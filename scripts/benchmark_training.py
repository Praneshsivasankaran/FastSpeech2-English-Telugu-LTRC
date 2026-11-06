#!/usr/bin/env python3
r"""Benchmark real training batches without writing a checkpoint or result artifact.

Example (run from the project directory on the target GPU):
    python scripts/benchmark_training.py --config English/config.yaml \
        --batch-sizes 8 16 --steps 20 --output /tmp/english-benchmark.json

Each candidate starts with the same fresh model and optimizer. Warm-up and a
conservative long-utterance stress step are reported separately from throughput.
CPU/MPS runs are useful smoke checks, but cannot qualify a CUDA batch size.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tts.modeling import (
    LOSS_NAMES, FastSpeech2Loss, build_model, device_for, read_config, read_vocab,
    resolve, seed_everything,
)
from tts.training import (
    Features, collate, epoch_batches, learning_rate_at_step, move,
    precision_context, synchronize,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--steps', type=int, default=20,
                        help='Measured representative training steps per candidate (default: 20).')
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True,
                        help='Benchmark JSON, outside configured checkpoint and result directories.')
    parser.add_argument('--device',
                        help='Override the configured device, for example cuda:0 or cpu.')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup_steps < 0 or any(b < 1 for b in args.batch_sizes):
        parser.error('steps and batch sizes must be positive; warmup-steps must be non-negative')
    if len(args.batch_sizes) != len(set(args.batch_sizes)):
        parser.error('batch-sizes must not contain duplicates')
    return args


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def environment_info(device=None):
    available = torch.cuda.is_available()
    info = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch_version': str(torch.__version__), 'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(), 'cuda_available': available,
        'selected_device': str(device) if device is not None else None,
        'cuda_execution': bool(device is not None and device.type == 'cuda' and available),
    }
    if available:
        info['gpu_devices'] = []
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            info['gpu_devices'].append({
                'index': index, 'name': prop.name,
                'compute_capability': [prop.major, prop.minor],
                'total_memory_bytes': prop.total_memory,
                'multiprocessor_count': prop.multi_processor_count,
            })
        info['selected_cuda_index'] = torch.cuda.current_device()
        info['bf16_supported'] = torch.cuda.is_bf16_supported()
    else:
        info['gpu_devices'] = []
        info['bf16_supported'] = False
    return info


def array_shape(archive, name):
    """Read only a NumPy header; avoid decompressing every full mel spectrogram."""
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NumPy header version {version}')
    if dtype.hasobject:
        raise ValueError('Object feature arrays are not permitted')
    return shape


def feature_lengths(dataset, max_frames):
    lengths = []
    for row in dataset.rows:
        with zipfile.ZipFile(resolve(row['features'])) as archive:
            mel = array_shape(archive, 'mel')
            phones = array_shape(archive, 'phones')
        if len(mel) != 2 or mel[1] != 80 or len(phones) != 1:
            raise ValueError(f'Invalid feature shape for {row["id"]}')
        if not (2 <= mel[0] <= max_frames and 1 <= phones[0] <= max_frames):
            raise ValueError(f'Feature exceeds configured positional limits for {row["id"]}')
        lengths.append((int(mel[0]), int(phones[0])))
    return lengths


def duration_seconds(rows):
    values = [row.get('duration_seconds', row.get('duration')) for row in rows]
    if any(value is None for value in values):
        return None
    values = [float(value) for value in values]
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Manifest duration metadata must be finite and positive')
    return float(sum(values))


def manifest_info(dataset):
    seconds = duration_seconds(dataset.rows)
    return {
        'path': str(dataset.path), 'examples': len(dataset),
        'manifest_sha256': hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
        'duration_seconds': seconds,
        'duration_hours': seconds / 3600 if seconds is not None else None,
    }


def schedule(rows, batch_size, steps, seed, bucket_size):
    if len(rows) < batch_size:
        raise ValueError(f'{len(rows)} real examples cannot fill batch size {batch_size}')
    result = []
    epoch = 0
    while len(result) < steps:
        result.extend(indices for indices in epoch_batches(rows, batch_size, seed, epoch, bucket_size)
                      if len(indices) == batch_size)
        epoch += 1
    return result[:steps]


def stress_indices(lengths, batch_size):
    """Full real batch containing both the longest mel and longest phone sequence."""
    selected = sorted(range(len(lengths)), key=lambda i: (lengths[i][0], lengths[i][1], i),
                      reverse=True)[:batch_size]
    longest_phones = max(range(len(lengths)), key=lambda i: (lengths[i][1], lengths[i][0], i))
    if longest_phones not in selected:
        # For batch size one, the longest-mel sample is tested first and a second
        # step tests the longest-phone sample when they are different utterances.
        if batch_size == 1:
            return [selected, [longest_phones]]
        selected[-1] = longest_phones
    return [selected]


def memory_info(device):
    if device.type != 'cuda':
        return {'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None}
    return {
        'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
    }


def timed_step(model, optimizer, lossfn, dataset, indices, device, precision, training, step):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    batch = collate([dataset[index] for index in indices])
    loaded = time.perf_counter()
    true_frames = int(batch[7].sum().item())
    true_phones = int(batch[4].sum().item())
    padded_frames = len(indices) * batch[8]
    padded_phones = len(indices) * batch[5]
    batch = move(batch, device)
    synchronize(device)
    transferred = time.perf_counter()
    peak_lr = training.get('peak_learning_rate', training.get('learning_rate', 1e-4))
    lr = learning_rate_at_step(step, peak_lr, training.get('warmup_steps', 0))
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    with precision_context(device, precision):
        predictions = model(*batch[2:])
        losses = lossfn(batch, predictions)
    if not torch.isfinite(losses[0]):
        raise FloatingPointError('Non-finite training loss')
    losses[0].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.get('grad_clip', 1.0))
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite gradient norm')
    optimizer.step()
    synchronize(device)
    finished = time.perf_counter()
    train_seconds = finished - transferred
    end_to_end = finished - start
    return {
        'example_ids': batch[0], 'examples': len(indices), 'true_mel_frames': true_frames,
        'padded_mel_frames': padded_frames, 'max_mel_frames': batch[8],
        'max_phones': batch[5], 'mel_padding_ratio': 1 - true_frames / padded_frames,
        'phone_padding_ratio': 1 - true_phones / padded_phones,
        'data_loading_collation_seconds': loaded - start,
        'device_transfer_seconds': transferred - loaded,
        'training_seconds': train_seconds, 'end_to_end_seconds': end_to_end,
        'true_mel_frames_per_second': true_frames / train_seconds,
        'end_to_end_true_mel_frames_per_second': true_frames / end_to_end,
        'learning_rate': lr, 'grad_norm': float(norm.item()),
        'losses': {name: float(value.item()) for name, value in zip(LOSS_NAMES, losses)},
        **memory_info(device),
    }


def aggregate(records):
    if not records:
        return None
    frames = sum(record['true_mel_frames'] for record in records)
    padded = sum(record['padded_mel_frames'] for record in records)
    seconds = sum(record['training_seconds'] for record in records)
    end_to_end = sum(record['end_to_end_seconds'] for record in records)
    return {
        'completed_steps': len(records), 'examples': sum(r['examples'] for r in records),
        'true_mel_frames': frames, 'training_seconds': seconds,
        'end_to_end_seconds': end_to_end,
        'median_training_step_seconds': float(np.median([r['training_seconds'] for r in records])),
        'true_mel_frames_per_second': frames / seconds,
        'end_to_end_true_mel_frames_per_second': frames / end_to_end,
        'mel_padding_ratio': 1 - frames / padded,
    }


def benchmark(config, vocab, dataset, lengths, batch_size, args, device, precision):
    training = config['training']
    seed = training.get('seed', 20261105)
    result = {
        'batch_size': batch_size, 'status': 'running', 'precision': precision,
        'seed': seed, 'warmup_steps_requested': args.warmup_steps,
        'measured_steps_requested': args.steps,
        'length_bucket_size': training.get('length_bucket_size', 0),
        'warmup': [], 'representative': [], 'long_sample_stress': [],
        'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None,
    }
    model = optimizer = lossfn = None
    stage = 'setup'
    indices = None
    began = time.perf_counter()
    try:
        batches = schedule(dataset.rows, batch_size, args.warmup_steps + args.steps,
                           seed, training.get('length_bucket_size', 0))
        stress = stress_indices(lengths, batch_size)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        seed_everything(seed)
        model = build_model(config, vocab, device)
        model.train()
        lossfn = FastSpeech2Loss(config['preprocess'], config['model'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.get('peak_learning_rate', training.get('learning_rate', 1e-4)),
            betas=(.9, .98), eps=1e-9, weight_decay=training.get('weight_decay', 0),
        )
        result['parameters'] = sum(parameter.numel() for parameter in model.parameters())
        plans = [('warmup', batches[:args.warmup_steps]),
                 ('representative', batches[args.warmup_steps:]),
                 ('long_sample_stress', stress)]
        step = 0
        for stage, stage_batches in plans:
            for stage_step, indices in enumerate(stage_batches, start=1):
                step += 1
                record = timed_step(model, optimizer, lossfn, dataset, indices, device,
                                    precision, training, step)
                record.update(step=step, stage_step=stage_step)
                result[stage].append(record)
                for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
                    value = record[key]
                    if value is not None:
                        result[key] = max(result[key] or 0, value)
                print(json.dumps({'batch_size': batch_size, 'stage': stage,
                                  'stage_step': stage_step,
                                  'training_seconds': record['training_seconds'],
                                  'true_mel_frames_per_second': record['true_mel_frames_per_second'],
                                  **{key: record[key] for key in ('peak_cuda_allocated_bytes',
                                                                 'peak_cuda_reserved_bytes')}}), flush=True)
        result['status'] = 'success'
    except Exception as exc:
        oom = isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower()
        result['status'] = 'oom' if oom else 'error'
        result['failure'] = {
            'stage': stage, 'error_type': type(exc).__name__, 'message': str(exc),
            'example_ids': [dataset.rows[i]['id'] for i in indices] if indices is not None else [],
            **memory_info(device),
        }
        for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
            value = result['failure'][key]
            if value is not None:
                result[key] = max(result[key] or 0, value)
    finally:
        # Do not retain an autograd graph or a failed candidate's allocator cache
        # while trying the next independently initialized model.
        del optimizer, lossfn, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result['wall_seconds'] = time.perf_counter() - began
    result['representative_summary'] = aggregate(result['representative'])
    result['stress_summary'] = aggregate(result['long_sample_stress'])
    result['cuda_batch_qualified'] = bool(result['status'] == 'success' and device.type == 'cuda')
    return result


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    report = {
        'format_version': 1, 'purpose': 'stateless_training_batch_benchmark',
        'status': 'running', 'config': str(args.config.expanduser().resolve()),
        'output': str(output), 'checkpoint_written': False, 'runs': [],
        'environment': environment_info(),
        'measurement': {
            'throughput_numerator': 'actual unpadded target mel frames',
            'training_time': 'synchronized forward, loss, backward, gradient clipping and optimizer step',
            'end_to_end_time': 'feature load, collation, device transfer and training',
            'mel_padding_ratio': '1 - actual mel frames / allocated batch mel-frame positions',
            'representative_selection': 'seeded training epoch length buckets, excluding incomplete batches',
            'stress_selection': 'longest actual mel sequences plus the longest phone sequence',
            'warmup_and_stress_excluded_from_representative_throughput': True,
            'precision_fallback': 'float32 if requested BF16 is unavailable on the selected device',
        },
    }
    try:
        config = read_config(args.config.expanduser().resolve())
        source_paths = [args.config.expanduser().resolve()]
        source_paths.extend(resolve(config['paths'][key]).resolve()
                            for key in ('vocab', 'train_manifest', 'val_manifest'))
        if output in source_paths:
            raise ValueError('Benchmark output must be outside configured input files')
        for name in ('checkpoint_dir', 'result_dir'):
            protected = resolve(config['paths'][name]).resolve()
            if output == protected or protected in output.parents:
                raise ValueError(f'Benchmark output must be outside configured {name}')
        training = config['training']
        device = device_for(args.device or training.get('device', 'auto'))
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no CUDA benchmark was performed')
            torch.cuda.set_device(device)
        if 'cpu_threads' in training:
            torch.set_num_threads(training['cpu_threads'])
        report['environment'] = environment_info(device)
        requested = training.get('precision', 'float32')
        if requested not in ('bf16', 'float32'):
            raise ValueError('precision must be float32 or bf16')
        supported = device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        precision = 'bf16' if requested == 'bf16' and supported else 'float32'
        report['precision_requested'] = requested
        report['precision_used'] = precision
        report['precision_fallback_reason'] = (
            'The selected device does not support CUDA BF16' if precision != requested else None)
        precision_context(device, precision)
        vocab = read_vocab(config)
        train = Features(config['paths']['train_manifest'], vocab, config['model']['max_seq_len'])
        val = Features(config['paths']['val_manifest'], vocab, config['model']['max_seq_len'])
        train_info, val_info = manifest_info(train), manifest_info(val)
        report['data'] = {'training': train_info, 'validation': val_info,
                          'source': 'configured feature manifests; no synthetic examples'}
        tr_seconds, va_seconds = train_info['duration_seconds'], val_info['duration_seconds']
        report['data']['approximately_5h_train_1h_validation'] = bool(
            tr_seconds is not None and va_seconds is not None
            and 16200 <= tr_seconds <= 19800 and 3240 <= va_seconds <= 3960)
        lengths = feature_lengths(train, config['model']['max_seq_len'])
        report['data']['training_mel_frame_percentiles'] = {
            str(percent): float(np.percentile([item[0] for item in lengths], percent))
            for percent in (0, 25, 50, 75, 95, 99, 100)
        }
        report['data']['maximum_training_phones'] = max(item[1] for item in lengths)
        write_report(output, report)
        for batch_size in args.batch_sizes:
            run = benchmark(config, vocab, train, lengths, batch_size, args, device, precision)
            run['cuda_execution_passed'] = run['cuda_batch_qualified']
            run['cuda_batch_qualified'] = bool(
                run['cuda_execution_passed'] and report['data']['approximately_5h_train_1h_validation'])
            report['runs'].append(run)
            write_report(output, report)
        successes = sum(run['status'] == 'success' for run in report['runs'])
        report['status'] = ('success' if successes == len(report['runs'])
                            else 'partial' if successes else 'failed')
        report['successful_batch_sizes'] = [run['batch_size'] for run in report['runs']
                                            if run['status'] == 'success']
        report['cuda_qualified_batch_sizes'] = [run['batch_size'] for run in report['runs']
                                                if run['cuda_batch_qualified']]
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = {'error_type': type(exc).__name__, 'message': str(exc)}
        # A protected output must never be written, including on error.
        if isinstance(exc, ValueError) and str(exc).startswith('Benchmark output must be outside'):
            print(json.dumps(report), flush=True)
            return 1
    write_report(output, report)
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'successful_batch_sizes': report.get('successful_batch_sizes', []),
                      'cuda_qualified_batch_sizes': report.get('cuda_qualified_batch_sizes', [])}), flush=True)
    return 0 if report['status'] in ('success', 'partial') else 1


if __name__ == '__main__':
    raise SystemExit(main())

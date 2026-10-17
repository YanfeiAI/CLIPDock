"""CLI parsing and VS-score batch-size planning; no reference tree needed."""

import pytest


def test_batch_size_argument():
    from clipdock.main import batch_size_arg

    assert batch_size_arg('auto') == 'auto'
    assert batch_size_arg('12') == 12
    assert batch_size_arg('1') == 1
    for bad in ('0', '-1', 'foo', '12.5'):
        with pytest.raises(Exception):
            batch_size_arg(bad)


def test_auto_batch_size_selector(monkeypatch):
    torch = pytest.importorskip('torch')
    from model.graph_score import AUTO_BATCH_CANDIDATES, DEFAULT_BATCH_SIZE, select_batch_size

    class Device:
        type = 'cuda'

    monkeypatch.setattr(torch.cuda, 'mem_get_info',
                        lambda device=None: (32 * 2 ** 30, 32 * 2 ** 30))
    assert select_batch_size('auto', Device(), 334) == max(AUTO_BATCH_CANDIDATES)
    assert select_batch_size(48, Device(), 334) == 48  # explicit values are untouched

    picks = {}
    for gb in (2, 4, 8, 16, 24, 32):
        monkeypatch.setattr(torch.cuda, 'mem_get_info',
                            lambda device=None, g=gb: (g * 2 ** 30, 32 * 2 ** 30))
        picks[gb] = select_batch_size('auto', Device(), 334)
    assert all(p in AUTO_BATCH_CANDIDATES for p in picks.values())
    assert picks[2] == DEFAULT_BATCH_SIZE
    assert list(picks.values()) == sorted(picks.values()), picks
    assert picks[8] == 24, picks


def test_auto_batch_size_cpu_fallback():
    torch = pytest.importorskip('torch')
    from model.graph_score import DEFAULT_BATCH_SIZE, select_batch_size

    assert select_batch_size('auto', torch.device('cpu'), 334) == DEFAULT_BATCH_SIZE

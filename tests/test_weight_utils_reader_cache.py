# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: a single model load must not re-parse the same GGUF
file's metadata from scratch for every helper that needs it.

get_gguf_tensor_names, get_gguf_unquantized_params, and
gguf_quant_weights_iterator_multi are all called on the same backbone file(s)
during a single _prepare_adapter() pass in loader.py. Each used to construct
its own fresh gguf.GGUFReader, which re-parses the entire KV metadata section
(including the full tokenizer vocab array) from raw bytes in pure Python --
wasted work repeated 3+ times per load. They should now share one cached
reader per file path instead.
"""

from unittest.mock import patch

import gguf
import numpy as np
import pytest

from vllm_gguf_plugin.weight_utils import (
    _cached_gguf_reader,
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)


@pytest.fixture
def minimal_gguf_file(tmp_path):
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def test_gguf_reader_is_shared_across_helpers(minimal_gguf_file):
    """The three helpers, called on the same file, should construct exactly
    one GGUFReader between them, not one each."""
    _cached_gguf_reader.cache_clear()

    with patch(
        "vllm_gguf_plugin.weight_utils.gguf.GGUFReader",
        wraps=gguf.GGUFReader,
    ) as spy:
        names = get_gguf_tensor_names([minimal_gguf_file])
        unquantized = get_gguf_unquantized_params([minimal_gguf_file])
        list(gguf_quant_weights_iterator_multi([minimal_gguf_file]))

    assert spy.call_count == 1, (
        f"expected a single shared GGUFReader construction across "
        f"get_gguf_tensor_names, get_gguf_unquantized_params, and "
        f"gguf_quant_weights_iterator_multi, but GGUFReader() was called "
        f"{spy.call_count} times"
    )
    assert names == {"token_embd.weight", "blk.0.attn_q.weight"}
    assert set(unquantized) == {"token_embd.weight", "blk.0.attn_q.weight"}


def test_gguf_reader_cache_keyed_by_path(minimal_gguf_file, tmp_path):
    """Different files must not share a cached reader."""
    _cached_gguf_reader.cache_clear()

    other_path = tmp_path / "other.gguf"
    writer = gguf.GGUFWriter(str(other_path), "llama")
    writer.add_tensor("token_embd.weight", np.zeros((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader_a = _cached_gguf_reader(minimal_gguf_file)
    reader_b = _cached_gguf_reader(str(other_path))
    reader_a_again = _cached_gguf_reader(minimal_gguf_file)

    assert reader_a is reader_a_again
    assert reader_a is not reader_b
    assert _cached_gguf_reader.cache_info().hits == 1
    assert _cached_gguf_reader.cache_info().misses == 2

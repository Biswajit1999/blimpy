from unittest.mock import patch

from blimpy.io import hdf_writer


def test_bitshuffle_options_uses_current_api():
    expected = {"compression": 32008, "compression_opts": (0, 2)}
    with patch.object(hdf_writer.hdf5plugin, "Bitshuffle", return_value=expected) as bitshuffle:
        result = hdf_writer._bitshuffle_options()

    assert result == expected
    bitshuffle.assert_called_once_with(nelems=0, cname="lz4")


def test_bitshuffle_options_falls_back_to_legacy_api():
    expected = {"compression": 32008, "compression_opts": (0, 2)}
    with patch.object(
        hdf_writer.hdf5plugin,
        "Bitshuffle",
        side_effect=[TypeError("unexpected keyword argument cname"), expected],
    ) as bitshuffle:
        result = hdf_writer._bitshuffle_options()

    assert result == expected
    assert bitshuffle.call_count == 2
    assert bitshuffle.call_args_list[0].kwargs == {"nelems": 0, "cname": "lz4"}
    assert bitshuffle.call_args_list[1].kwargs == {"nelems": 0, "lz4": True}

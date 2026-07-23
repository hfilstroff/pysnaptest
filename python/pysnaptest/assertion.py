"""Snapshot assertion helpers.

This module wraps the Rust snapshot implementation used by ``pysnaptest`` and
provides Python friendly helpers for asserting snapshots of common data
structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, overload
from functools import wraps
import asyncio

from ._pysnaptest import (
    assert_json_snapshot as _assert_json_snapshot,
    assert_csv_snapshot as _assert_csv_snapshot,
    assert_snapshot as _assert_snapshot,
    assert_binary_snapshot as _assert_binary_snapshot,
    assert_binary_snapshot_capturing_previous as _assert_binary_snapshot_capturing_previous,
    render_text_diff as _render_text_diff,
    SnapshotInfo,
)
from .encoders import is_jsonable_object, to_jsonable

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def sorted_redaction() -> None:
    """Mark a list for sorting before snapshot comparison.

    Returns:
        None: A sentinel value recognised by the snapshot machinery.
    """

    return None


def rounded_redaction(decimals: int) -> int:
    """Round numbers before snapshotting.

    Args:
        decimals: Number of decimal places to round to.

    Returns:
        int: The ``decimals`` argument, passed through.
    """

    return decimals


def extract_from_pytest_env(
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    allow_duplicates: bool = False,
) -> SnapshotInfo:
    """Load snapshot info from the active pytest test.

    Args:
        snapshot_path: Optional path override for storing snapshots.
        snapshot_name: Optional name override for the snapshot file.
        allow_duplicates: Whether to allow duplicate snapshot names.

    Returns:
        SnapshotInfo: Snapshot configuration for the active test.
    """

    return SnapshotInfo.from_pytest(
        snapshot_path_override=snapshot_path,
        snapshot_name_override=snapshot_name,
        allow_duplicates=allow_duplicates,
    )


def assert_json_snapshot(
    result: Any,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    allow_duplicates: bool = False,
    custom_encoder: Optional[Dict[type, Callable[[Any], Any]]] = None,
) -> None:
    """Assert that a value matches a stored JSON snapshot.

    The ``result`` is normalized with :func:`pysnaptest.to_jsonable` before being
    serialized, so Pydantic models, dataclasses, enums and common
    standard-library types (``datetime``, ``UUID``, ``Decimal``, ...) are
    supported automatically.

    Args:
        result: Object that will be serialized to JSON.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        allow_duplicates: Whether to allow duplicate snapshot names.
        custom_encoder: Optional mapping of types to encoder callables used when
            normalizing ``result``.

    Raises:
        TypeError: If ``result`` is a pandas or polars ``DataFrame``. Use
            :func:`assert_dataframe_snapshot` instead.
    """

    if try_is_pandas_df(result) or try_is_polars_df(result):
        raise TypeError(
            "DataFrames are not supported by assert_json_snapshot. Use "
            "assert_dataframe_snapshot(df, dataframe_snapshot_format='json') instead."
        )

    result = to_jsonable(result, custom_encoder=custom_encoder)
    test_info = extract_from_pytest_env(snapshot_path, snapshot_name, allow_duplicates)
    _assert_json_snapshot(test_info, result, redactions)


def assert_csv_snapshot(
    result: Any,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    allow_duplicates: bool = False,
) -> None:
    """Assert that CSV text matches the stored snapshot.

    Args:
        result: CSV string to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        allow_duplicates: Whether to allow duplicate snapshot names.
    """

    test_info = extract_from_pytest_env(snapshot_path, snapshot_name, allow_duplicates)
    _assert_csv_snapshot(test_info, result, redactions)


def try_is_pandas_df(maybe_df: Any) -> bool:
    """Check whether an object appears to be a pandas ``DataFrame``.

    Args:
        maybe_df: Object to test.

    Returns:
        bool: ``True`` if ``maybe_df`` is a pandas ``DataFrame``.
    """

    try:
        import pandas as pd
    except ImportError:
        return False

    return isinstance(maybe_df, pd.DataFrame)


def try_is_polars_df(maybe_df: Any) -> bool:
    """Check whether an object appears to be a polars ``DataFrame``.

    Args:
        maybe_df: Object to test.

    Returns:
        bool: ``True`` if ``maybe_df`` is a polars ``DataFrame``.
    """

    try:
        import polars as pl
    except ImportError:
        return False

    return isinstance(maybe_df, pl.DataFrame)


def _df_to_readable(df: Any, readable_diff: str) -> str:
    """Render a DataFrame to deterministic CSV or JSON text for diffing.

    Args:
        df: A pandas or polars ``DataFrame``.
        readable_diff: ``"csv"`` or ``"json"``.

    Returns:
        str: A line-oriented text rendering (one row per line for CSV, an
        indented list of row objects for JSON) suitable for a unified diff.

    Raises:
        ValueError: If ``readable_diff`` is not ``"csv"`` or ``"json"``.
    """

    if readable_diff == "csv":
        if try_is_pandas_df(df):
            return df.to_csv(index=False)
        return df.write_csv()
    if readable_diff == "json":
        import json

        if try_is_pandas_df(df):
            rows = df.to_dict(orient="records")
        else:
            rows = df.to_dicts()
        return json.dumps(rows, indent=2, default=str, ensure_ascii=False)
    raise ValueError(
        f"Unsupported readable_diff format: {readable_diff!r}. Use 'csv' or 'json'."
    )


def _decode_binary_dataframe(previous: bytes, df: Any, extension: str) -> Any:
    """Reconstruct a DataFrame from the committed binary snapshot bytes.

    The committed bytes are decoded with the same library as ``df`` (so the
    rendered diff is apples-to-apples) and the same format used to store them.

    Args:
        previous: The committed sidecar bytes.
        df: The current DataFrame (used only to pick pandas vs polars).
        extension: The stored binary format (``"parquet"`` or ``"bin"``).

    Returns:
        Any: The decoded pandas or polars ``DataFrame``.
    """

    import io

    buffer = io.BytesIO(previous)
    if try_is_pandas_df(df):
        import pandas as pd

        return pd.read_parquet(buffer)

    import polars as pl

    if extension == "bin":
        return pl.DataFrame.deserialize(buffer, format="binary")
    return pl.read_parquet(buffer)


def _assert_binary_dataframe_snapshot(
    df: Any,
    result: bytes,
    extension: str,
    snapshot_path: Optional[str],
    snapshot_name: Optional[str],
    allow_duplicates: bool,
    readable_diff: Optional[str],
) -> None:
    """Assert a DataFrame's binary snapshot, showing a readable diff on mismatch.

    Equality is insta's exact byte comparison of the stored binary (e.g.
    parquet). When ``readable_diff`` is ``None`` this behaves exactly like
    :func:`assert_binary_snapshot`. Otherwise, on a byte mismatch the committed
    snapshot is decompressed and a CSV/JSON unified diff (rendered with insta's
    own diff engine) is attached to the raised ``AssertionError``.

    Args:
        df: The DataFrame being snapshotted.
        result: The serialized binary bytes to store and compare.
        extension: The binary format extension (``"parquet"`` or ``"bin"``).
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        allow_duplicates: Whether to allow duplicate snapshot names.
        readable_diff: ``None`` (byte-only), ``"csv"`` or ``"json"``.

    Raises:
        AssertionError: If the snapshot does not match, with a readable diff
            appended when ``readable_diff`` is set.
    """

    test_info = extract_from_pytest_env(snapshot_path, snapshot_name, allow_duplicates)

    if readable_diff is None:
        _assert_binary_snapshot(test_info, extension, result)
        return

    previous = _assert_binary_snapshot_capturing_previous(test_info, extension, result)
    if previous is None:
        return

    base_message = (
        "DataFrame snapshot did not match the stored value (readable diff below). "
        "Update the snapshot if this change is intentional."
    )
    if not previous:
        raise AssertionError(base_message)

    old_df = _decode_binary_dataframe(previous, df, extension)
    diff = _render_text_diff(
        _df_to_readable(old_df, readable_diff),
        _df_to_readable(df, readable_diff),
        "committed",
        "new",
    )
    raise AssertionError(f"{base_message}\n\n{diff}")


def assert_pandas_dataframe_snapshot(
    df: pd.DataFrame,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    readable_diff: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
    """Snapshot assertion for pandas DataFrames.

    Args:
        df: The DataFrame to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        dataframe_snapshot_format: One of ``"csv"``, ``"json"`` or ``"parquet"``.
        allow_duplicates: Whether to allow duplicate snapshot names.
        readable_diff: For the binary ``"parquet"`` format only, show a readable
            ``"csv"`` or ``"json"`` diff on mismatch instead of just reporting a
            byte difference. ``None`` (default) keeps the byte-only behavior.
        *args: Positional arguments forwarded to the DataFrame export method.
        **kwargs: Keyword arguments forwarded to the DataFrame export method.
    """

    if dataframe_snapshot_format == "csv":
        result = df.to_csv(*args, **kwargs)
        assert_csv_snapshot(
            result, snapshot_path, snapshot_name, redactions, allow_duplicates
        )
    elif dataframe_snapshot_format == "json":
        result = df.to_dict(orient="list", *args, **kwargs)
        assert_json_snapshot(
            result, snapshot_path, snapshot_name, redactions, allow_duplicates
        )
    elif dataframe_snapshot_format == "parquet":
        result = df.to_parquet(engine="pyarrow")
        _assert_binary_dataframe_snapshot(
            df,
            result,
            dataframe_snapshot_format,
            snapshot_path,
            snapshot_name,
            allow_duplicates,
            readable_diff,
        )
    else:
        raise ValueError(
            "Unsupported snapshot format for dataframes, supported formats are: 'csv', 'json', 'parquet'."
        )


def assert_polars_dataframe_snapshot(
    df: pl.DataFrame,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    readable_diff: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
    """Snapshot assertion for polars DataFrames.

    Args:
        df: The DataFrame to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        dataframe_snapshot_format: One of ``"csv"``, ``"json"`` or ``"bin"``.
        allow_duplicates: Whether to allow duplicate snapshot names.
        readable_diff: For the binary ``"bin"`` format only, show a readable
            ``"csv"`` or ``"json"`` diff on mismatch instead of just reporting a
            byte difference. ``None`` (default) keeps the byte-only behavior.
        *args: Positional arguments forwarded to the DataFrame export method.
        **kwargs: Keyword arguments forwarded to the DataFrame export method.
    """

    if dataframe_snapshot_format == "csv":
        result = df.write_csv(*args, **kwargs)
        assert_csv_snapshot(
            result, snapshot_path, snapshot_name, redactions, allow_duplicates
        )
    elif dataframe_snapshot_format == "json":
        result = df.to_dict(as_series=False)
        assert_json_snapshot(
            result, snapshot_path, snapshot_name, redactions, allow_duplicates
        )
    elif dataframe_snapshot_format == "bin":
        result = df.serialize(format="binary", *args, **kwargs)
        _assert_binary_dataframe_snapshot(
            df,
            result,
            dataframe_snapshot_format,
            snapshot_path,
            snapshot_name,
            allow_duplicates,
            readable_diff,
        )
    else:
        raise ValueError(
            "Unsupported snapshot format for polars dataframes, supported formats are: 'csv', 'json', 'bin'."
        )


def assert_dataframe_snapshot(
    df: Union[pd.DataFrame, pl.DataFrame],
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    readable_diff: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
    """Snapshot assertion for either pandas or polars ``DataFrame`` objects.

    Args:
        df: The DataFrame to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        dataframe_snapshot_format: Format to serialize the DataFrame as. Supported
            values are ``"csv"``, ``"json"``, ``"parquet"`` and ``"bin"``.
        allow_duplicates: Whether to allow duplicate snapshot names.
        readable_diff: For the binary formats (``"parquet"``/``"bin"``) only, show
            a readable ``"csv"`` or ``"json"`` diff on mismatch instead of just a
            byte difference. ``None`` (default) keeps the byte-only behavior.
        *args: Positional arguments forwarded to the DataFrame export method.
        **kwargs: Keyword arguments forwarded to the DataFrame export method.
    """

    if try_is_pandas_df(df):
        assert_pandas_dataframe_snapshot(
            df,
            snapshot_path,
            snapshot_name,
            redactions,
            dataframe_snapshot_format,
            allow_duplicates,
            readable_diff,
            *args,
            **kwargs,
        )
    elif try_is_polars_df(df):
        assert_polars_dataframe_snapshot(
            df,
            snapshot_path,
            snapshot_name,
            redactions,
            dataframe_snapshot_format,
            allow_duplicates,
            readable_diff,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(
            "Unsupported dataframe type, only pandas and polars are supported. (We may also be unable to import both pandas and polars for some reason, but this is not likely)"
        )


def assert_binary_snapshot(
    result: bytes,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    extension: str = "bin",
    allow_duplicates: bool = False,
) -> None:
    """Assert that binary data matches the stored snapshot.

    Args:
        result: Raw bytes to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        extension: File extension to use when saving the snapshot.
        allow_duplicates: Whether to allow duplicate snapshot names.
    """

    test_info = extract_from_pytest_env(snapshot_path, snapshot_name, allow_duplicates)
    _assert_binary_snapshot(test_info, extension, result)


def assert_snapshot(
    result: Any,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    allow_duplicates: bool = False,
) -> None:
    """Assert that a string matches the stored snapshot.

    Args:
        result: Text to snapshot.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        allow_duplicates: Whether to allow duplicate snapshot names.
    """

    test_info = extract_from_pytest_env(snapshot_path, snapshot_name, allow_duplicates)
    _assert_snapshot(test_info, result)


def insta_snapshot(
    result: Any,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    custom_encoder: Optional[Dict[type, Callable[[Any], Any]]] = None,
    readable_diff: Optional[str] = None,
) -> None:
    """Dispatch a value to the appropriate snapshot assertion.

    Args:
        result: Value to snapshot. Supported types include ``dict``, ``list``,
            ``bytes``, pandas or polars ``DataFrame`` objects, and any object
            recognised by :func:`pysnaptest.to_jsonable` (Pydantic models,
            dataclasses, enums, sets, tuples and mappings).
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        dataframe_snapshot_format: Format used when snapshotting DataFrames.
        allow_duplicates: Whether to allow duplicate snapshot names.
        custom_encoder: Optional mapping of types to encoder callables used when
            normalizing JSON snapshots.
        readable_diff: For binary DataFrame formats, show a ``"csv"``/``"json"``
            diff on mismatch. ``None`` (default) keeps byte-only reporting.
    """

    if isinstance(result, (dict, list)):
        assert_json_snapshot(
            result,
            snapshot_path,
            snapshot_name,
            redactions,
            allow_duplicates,
            custom_encoder=custom_encoder,
        )
    elif isinstance(result, bytes):
        assert_binary_snapshot(
            result,
            snapshot_path,
            snapshot_name,
            allow_duplicates=allow_duplicates,
        )
    elif try_is_pandas_df(result) or try_is_polars_df(result):
        assert_dataframe_snapshot(
            result,
            snapshot_path,
            snapshot_name,
            redactions,
            dataframe_snapshot_format,
            allow_duplicates,
            readable_diff,
        )
    elif is_jsonable_object(result):
        assert_json_snapshot(
            result,
            snapshot_path,
            snapshot_name,
            redactions,
            allow_duplicates,
            custom_encoder=custom_encoder,
        )
    else:
        if redactions is not None:
            raise ValueError("Redactions may only be used with json or csv snapshots.")
        assert_snapshot(
            result,
            snapshot_path,
            snapshot_name,
            allow_duplicates=allow_duplicates,
        )


@overload
def snapshot(func: Callable) -> Callable: ...


@overload
def snapshot(
    *,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    custom_encoder: Optional[Dict[type, Callable[[Any], Any]]] = None,
    readable_diff: Optional[str] = None,
) -> Callable:  # noqa: F811
    ...


def snapshot(  # noqa: F811
    func: Optional[Callable] = None,
    *,
    snapshot_path: Optional[str] = None,
    snapshot_name: Optional[str] = None,
    redactions: Optional[Dict[str, Union[str, int, None]]] = None,
    dataframe_snapshot_format: str = "csv",
    allow_duplicates: bool = False,
    custom_encoder: Optional[Dict[type, Callable[[Any], Any]]] = None,
    readable_diff: Optional[str] = None,
) -> Callable:
    """Decorator that snapshots the return value of ``func``.

    Args:
        func: The function being decorated.
        snapshot_path: Optional path override for storing the snapshot.
        snapshot_name: Optional name override for the snapshot file.
        redactions: Mapping of selectors to replacement values.
        dataframe_snapshot_format: Format used when snapshotting DataFrames.
        allow_duplicates: Whether to allow duplicate snapshot names.
        custom_encoder: Optional mapping of types to encoder callables used when
            normalizing JSON snapshots.
        readable_diff: For binary DataFrame formats (``"parquet"``/``"bin"``),
            show a readable ``"csv"``/``"json"`` diff on mismatch instead of just
            a byte difference. ``None`` (default) keeps byte-only reporting.

    Returns:
        Callable: The wrapped function.
    """

    def _wrap(target: Callable) -> Callable:
        if not callable(target):
            raise TypeError("Not a callable. Did you use a non-keyword argument?")

        if asyncio.iscoroutinefunction(target):

            @wraps(target)
            async def asserted_func(*args: Any, **kwargs: Any):
                result = await target(*args, **kwargs)
                insta_snapshot(
                    result,
                    snapshot_path=snapshot_path,
                    snapshot_name=snapshot_name,
                    redactions=redactions,
                    dataframe_snapshot_format=dataframe_snapshot_format,
                    allow_duplicates=allow_duplicates,
                    custom_encoder=custom_encoder,
                    readable_diff=readable_diff,
                )

            return asserted_func

        @wraps(target)
        def asserted_func(*args: Any, **kwargs: Any):
            result = target(*args, **kwargs)
            insta_snapshot(
                result,
                snapshot_path=snapshot_path,
                snapshot_name=snapshot_name,
                redactions=redactions,
                dataframe_snapshot_format=dataframe_snapshot_format,
                allow_duplicates=allow_duplicates,
                custom_encoder=custom_encoder,
                readable_diff=readable_diff,
            )

        return asserted_func

    if func is not None:
        return _wrap(func)

    return _wrap

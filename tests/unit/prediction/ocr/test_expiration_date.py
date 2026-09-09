import datetime

import pytest
from pytest_mock import MockerFixture

from robotoff.prediction.ocr.expiration_date import (
    MAX_YEARS_IN_FUTURE,
    MAX_YEARS_IN_PAST,
    find_expiration_date,
    is_plausible_expiration_date,
)
from robotoff.types import PredictionType

TODAY = datetime.date(2026, 7, 24)


@pytest.mark.parametrize(
    "date,expected",
    [
        # in-window dates are plausible
        (datetime.date(2026, 6, 14), True),
        (datetime.date(2027, 1, 1), True),
        (TODAY, True),
        # boundaries (inclusive)
        (datetime.date(TODAY.year + MAX_YEARS_IN_FUTURE, 1, 1), True),
        (datetime.date(TODAY.year - MAX_YEARS_IN_PAST, 12, 31), True),
        # just outside the window
        (datetime.date(TODAY.year + MAX_YEARS_IN_FUTURE + 1, 1, 1), False),
        (datetime.date(TODAY.year - MAX_YEARS_IN_PAST - 1, 12, 31), False),
    ],
)
def test_is_plausible_expiration_date(date, expected):
    assert is_plausible_expiration_date(date, today=TODAY) is expected


def test_is_plausible_expiration_date_defaults_to_today():
    # A date in the current year must always be plausible, whatever the year the
    # test runs in (guards against the previous hardcoded-window regression).
    today = datetime.date.today()
    assert is_plausible_expiration_date(datetime.date(today.year, 1, 1)) is True


def test_find_expiration_date_keeps_current_and_future_years():
    # Regression test for the hardcoded [2015, 2025] window that silently
    # dropped every expiration date from 2026 onwards. Build the year relative
    # to the current date so this test stays valid in future years.
    next_year = datetime.date.today().year + 1
    predictions = find_expiration_date(
        f"À consommer de préférence avant 14.06.{next_year}"
    )
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.type == PredictionType.expiration_date
    # value is normalized to ISO 8601
    assert prediction.value == f"{next_year}-06-14"


def test_find_expiration_date_drops_implausible_years():
    # A year far in the past is OCR noise and must be discarded.
    old_year = datetime.date.today().year - (MAX_YEARS_IN_PAST + 10)
    assert find_expiration_date(f"lot 14.06.{old_year}") == []


@pytest.fixture
def fixed_today(mocker: MockerFixture) -> datetime.date:
    # Freeze only the predictor's clock; keep real calendar-date parsing.
    clock = mocker.patch(
        "robotoff.prediction.ocr.expiration_date.datetime", wraps=datetime
    )
    clock.date.today.return_value = TODAY
    return TODAY


@pytest.mark.usefixtures("fixed_today")
@pytest.mark.parametrize("separator", ["-", ".", "/"])
@pytest.mark.parametrize(
    "year,matcher_type",
    [("26", "full_digits_short"), ("2026", "full_digits_long")],
)
def test_find_expiration_date_supported_formats(
    separator: str, year: str, matcher_type: str
) -> None:
    raw = separator.join(("14", "06", year))
    predictions = find_expiration_date(f"Best before {raw}")

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.type == PredictionType.expiration_date
    assert prediction.value == "2026-06-14"
    assert prediction.data == {"raw": raw, "type": matcher_type}
    assert prediction.automatic_processing is True
    assert prediction.predictor == "regex"


@pytest.mark.parametrize("year_format", ["%y", "%Y"])
@pytest.mark.parametrize(
    "year_offset,month,day,expected",
    [
        (-MAX_YEARS_IN_PAST, 1, 1, True),
        (-MAX_YEARS_IN_PAST, 12, 31, True),
        (MAX_YEARS_IN_FUTURE, 1, 1, True),
        (MAX_YEARS_IN_FUTURE, 12, 31, True),
        (-MAX_YEARS_IN_PAST - 1, 12, 31, False),
        (MAX_YEARS_IN_FUTURE + 1, 1, 1, False),
    ],
)
def test_find_expiration_date_window_boundaries(
    fixed_today: datetime.date,
    year_format: str,
    year_offset: int,
    month: int,
    day: int,
    expected: bool,
) -> None:
    candidate = datetime.date(fixed_today.year + year_offset, month, day)
    raw = candidate.strftime(f"%d/%m/{year_format}")
    predictions = find_expiration_date(f"Best before {raw}")

    assert [prediction.value for prediction in predictions] == (
        [candidate.isoformat()] if expected else []
    )


@pytest.mark.usefixtures("fixed_today")
@pytest.mark.parametrize("year", ["26", "2026"])
@pytest.mark.parametrize("day_month", ["31/04", "00/01", "01/13", "29/02"])
def test_find_expiration_date_rejects_invalid_calendar_dates(
    year: str, day_month: str
) -> None:
    assert find_expiration_date(f"Best before {day_month}/{year}") == []


@pytest.mark.usefixtures("fixed_today")
@pytest.mark.parametrize("year", ["28", "2028"])
def test_find_expiration_date_accepts_leap_day(year: str) -> None:
    predictions = find_expiration_date(f"Best before 29/02/{year}")
    assert [prediction.value for prediction in predictions] == ["2028-02-29"]


@pytest.mark.parametrize(
    "raw,before,after",
    [
        ("01/01/2042", [], ["2042-01-01"]),
        ("31/12/2021", ["2021-12-31"], []),
    ],
)
def test_find_expiration_date_window_advances_with_clock(
    mocker: MockerFixture, raw: str, before: list[str], after: list[str]
) -> None:
    # A long-running worker must use the new year without a module reload.
    clock = mocker.patch(
        "robotoff.prediction.ocr.expiration_date.datetime", wraps=datetime
    )
    clock.date.today.side_effect = [
        datetime.date(2026, 12, 31),
        datetime.date(2027, 1, 1),
    ]

    assert [prediction.value for prediction in find_expiration_date(raw)] == before
    assert [prediction.value for prediction in find_expiration_date(raw)] == after

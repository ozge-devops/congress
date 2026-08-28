from vesta.vlm_parse import parse_deplot, parse_matcha_pair, parse_number


def test_deplot_bar_price_down():
    cap = "TITLE |\n bar | price \n 0 | 9768 \n 5 | 9656 \n 35 | 9325"
    assert parse_deplot(cap) == 0


def test_deplot_swapped_header_still_reads_price():
    cap = "TITLE |\n price | bar \n 0.00 | 9500 \n 5.00 | 9481 \n 30.00 | 9732"
    assert parse_deplot(cap) == 1


def test_deplot_too_short_is_none():
    assert parse_deplot("TITLE | bar | price") is None


def test_matcha_pair():
    assert parse_matcha_pair("first 100", "last 110") == 1
    assert parse_matcha_pair("9400", "9100") == 0
    assert parse_matcha_pair("n/a", "n/a") is None
    assert parse_number("about 9768") == 9768.0


if __name__ == "__main__":
    test_deplot_bar_price_down()
    test_deplot_swapped_header_still_reads_price()
    test_deplot_too_short_is_none()
    test_matcha_pair()
    print("vlm parse ok")

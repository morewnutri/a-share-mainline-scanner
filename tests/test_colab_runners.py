from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_colab_runners_do_not_mount_google_drive():
    scan = (ROOT / "colab" / "run_scan.py").read_text(encoding="utf-8")
    valuation = (ROOT / "colab" / "run_valuation.py").read_text(encoding="utf-8")
    root_valuation = (ROOT / "run_valuation.py").read_text(encoding="utf-8")
    for runner in (scan, valuation, root_valuation):
        assert "google.colab import drive" not in runner
        assert "drive.mount" not in runner


def test_colab_scan_displays_sideways_seed_table_and_chart():
    scan = (ROOT / "colab" / "run_scan.py").read_text(encoding="utf-8")
    assert "横盘火种（低位箱体）Top 30" in scan
    assert "sideways_seed_status" in scan
    assert "横盘火种雷达.png" in scan

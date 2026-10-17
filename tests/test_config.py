import re

from docqa.config import ROOT


def test_default_database_url_matches_docker_compose_port():
    """Regression: the compose file and the config default must publish/use the same port."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    port = re.search(r'"127\.0\.0\.1:(\d+):5432"', compose).group(1)
    assert f":{port}/docqa" in (ROOT / "src" / "docqa" / "config.py").read_text(encoding="utf-8")
    assert f":{port}/docqa" in (ROOT / ".env.example").read_text(encoding="utf-8")

from jubal.config import JubalConfig


def test_defaults_load_with_empty_dict():
    config = JubalConfig.from_dict({})
    assert config.voiceid.threshold == 0.72
    assert config.gate.verification_timeout_s == 2.5
    assert config.agent.endpoint_url.startswith("http")


def test_partial_override_keeps_other_defaults():
    config = JubalConfig.from_dict({"voiceid": {"threshold": 0.9, "backend": "sherpa"}})
    assert config.voiceid.threshold == 0.9
    assert config.voiceid.backend == "sherpa"
    assert config.voiceid.margin == 0.05  # untouched default


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("voiceid:\n  threshold: 0.8\n")
    config = JubalConfig.load(path)
    assert config.voiceid.threshold == 0.8

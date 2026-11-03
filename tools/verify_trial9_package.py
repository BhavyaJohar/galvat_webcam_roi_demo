#!/usr/bin/env python3
"""Check the supplied Trial9 exports and create CPU decoder reference fixtures.

No camera, training, conversion, accuracy evaluation or headset-FPS claim.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"nano": "Trial9NanoConfig.json", "small": "Trial9Config.json"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_profile(config):
    classes = config["classes"]
    names, mapping = config["modelClassNames"], config["modelToCanonicalIds"]
    if len(classes) != 45 or len(names) != 45 or len(mapping) != 45:
        raise ValueError("45 classes, model names and mapping entries required")
    ids = [c["canonicalId"] for c in classes]
    if any(type(i) is not int for i in ids + mapping) or sorted(ids) != list(range(45)):
        raise ValueError("canonical IDs must cover 0..44")
    if sorted(mapping) != list(range(45)) or len(set(names)) != 45:
        raise ValueError("mapping and names must be unique permutations")
    indexed = {c["canonicalId"]: c for c in classes}
    for c in classes:
        t = c["threshold"]
        if type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("invalid class threshold")
        if not c["name"] or not c["displayName"]:
            raise ValueError("class names must be nonempty")
    if any(names[i] != indexed[mapping[i]]["name"] for i in range(45)):
        raise ValueError("model/canonical class-name mismatch")
    if names[20] != "measuring_spoon" or mapping[20] != 44:
        raise ValueError("measuring spoon must map model20 to canonical44")
    # The original small profile has no export metadata; its exact binary and
    # configuration are pinned separately in the release manifest.
    if config.get("applyAdditionalNms", False) is not False or config.get("referenceIou", .65) != .65:
        raise ValueError("expected end-to-end no-extra-NMS with reference IoU .65")
    if "modelSha256" in config and len(config["modelSha256"]) != 64:
        raise ValueError("model SHA256 is required")
    return indexed


def threshold_fixtures(config, np):
    indexed = validate_profile(config)
    rows = []
    for model_id, canonical in enumerate(config["modelToCanonicalIds"]):
        entry = indexed[canonical]
        threshold = np.float32(entry["threshold"])
        for kind, confidence in (("below", np.nextafter(threshold, np.float32(-np.inf))),
                                 ("equal", threshold),
                                 ("above", np.nextafter(threshold, np.float32(np.inf)))):
            rows.append({"row": [10, 20, 30, 40, float(confidence), model_id],
                         "canonical_id": canonical, "name": entry["name"],
                         "boundary": kind, "keep": bool(confidence >= threshold)})
    return rows


def compare_candidate_map(candidate, config):
    """Name-only comparison. It cannot authenticate an unseen model's IDs."""
    indexed = validate_profile(config)
    rows = candidate["classes"]
    ids = [r["id"] for r in rows]
    names = [r["name"] for r in rows]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(rows))) or len(set(names)) != len(rows):
        raise ValueError("candidate map IDs/names must be unique")
    canonical = {r["name"]: i for i, r in indexed.items()}
    model_ids = {name: i for i, name in enumerate(config["modelClassNames"])}
    return {"identity": candidate["identity"], "model_identity_verified": False,
            "candidate_class_count": len(rows), "shared_class_count": len(set(names) & set(canonical)),
            "trial9_only_classes": sorted(set(canonical) - set(names)),
            "candidate_only_classes": sorted(set(names) - set(canonical)),
            "shared_mapping_by_name": [{"candidate_id": r["id"], "name": r["name"],
                                         "trial9_model_id": model_ids[r["name"]],
                                         "galvat_canonical_id": canonical[r["name"]]}
                                        for r in rows if r["name"] in canonical]}


def verify(models_dir, config_dir):
    import numpy as np
    import onnx
    import onnxruntime as ort
    result = {"schema": "galvat.trial9_package_check.v1",
              "platform": platform.platform(),
              "versions": {"numpy": np.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "available_providers": ort.get_available_providers(), "models": {},
              "limitations": ["Synthetic zero-input fixtures are not recognition/recall tests.",
                              "CPU ORT is not Unity/Sentis or Magic Leap timing.",
                              "RGB normalization and coordinate origin still need Unity fixture parity.",
                              "Validation does not link the supplied training checkpoint to these binaries."]}
    canonical = None
    release = json.loads((config_dir / "release_manifest.json").read_text())
    result["source_commit"] = release["source_commit"]
    for label, filename in PROFILES.items():
        profile_path = config_dir / filename
        config = json.loads(profile_path.read_text())
        indexed = validate_profile(config)
        ordered = [indexed[i]["name"] for i in range(45)]
        if canonical is not None and canonical != ordered:
            raise ValueError("nano and small canonical order differs")
        canonical = ordered
        identity = release["models"][label]
        if sha256(profile_path) != identity["config_sha256"]:
            raise ValueError(f"{label} configuration SHA256 mismatch")
        if config.get("modelAssetName", identity["asset_name"]) != identity["asset_name"]:
            raise ValueError(f"{label} model asset/profile mismatch")
        path = models_dir / (identity["asset_name"] + ".onnx")
        if sha256(path) != identity["model_sha256"] or config.get("modelSha256", sha256(path)) != sha256(path):
            raise ValueError(f"{label} model SHA256 mismatch")
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        names = ast.literal_eval({p.key: p.value for p in model.metadata_props}["names"])
        if list(names) != list(range(45)) or [names[i] for i in range(45)] != config["modelClassNames"]:
            raise ValueError("embedded ONNX class IDs differ from profile")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != "tensor(float)":
            raise ValueError("expected one float32 NCHW640 input")
        if len(outputs) != 1 or outputs[0].shape != [1, 40, 6] or outputs[0].type != "tensor(float)":
            raise ValueError("expected one float32 [1,40,6] output")
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        start = time.perf_counter()
        output = session.run(None, {inputs[0].name: tensor})[0]
        smoke_ms = (time.perf_counter() - start) * 1000
        if output.shape != (1, 40, 6) or not np.isfinite(output).all():
            raise ValueError("invalid output values/shape")
        scores, ids = output[0, :, 4], output[0, :, 5]
        if not ((scores >= 0).all() and (scores <= 1).all() and
                (ids == np.floor(ids)).all() and (ids >= 0).all() and (ids < 45).all()):
            raise ValueError("invalid confidence/class IDs")
        result["models"][label] = {
            "model_sha256": sha256(path), "config_sha256": sha256(profile_path),
            "model_asset": path.name, "config_asset": filename,
            "opset": [i.version for i in model.opset_import], "providers": session.get_providers(),
            "input_name": inputs[0].name, "output_name": outputs[0].name,
            "input_shape": inputs[0].shape, "output_shape": outputs[0].shape,
            "canonical_names": ordered, "model_names": config["modelClassNames"],
            "model_to_canonical": config["modelToCanonicalIds"],
            "input_fixture": {"generator": "all-zero float32 NCHW tensor", "sha256": hashlib.sha256(tensor.tobytes()).hexdigest()},
            "output_fixture": output.tolist(), "threshold_fixtures": threshold_fixtures(config, np),
            "first_call_smoke_ms_not_benchmark": smoke_ms,
            "structure_and_contract_passed": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config/trial9")
    parser.add_argument("--output", required=True, type=Path, help="new JSON path; never overwrite")
    parser.add_argument("--candidate-map", type=Path, help="optional unverified comparison class map")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = verify(args.models_dir, args.config_dir)
    if args.candidate_map:
        result["candidate_class_comparison"] = compare_candidate_map(
            json.loads(args.candidate_map.read_text()),
            json.loads((args.config_dir / PROFILES["small"]).read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS: both ONNX hashes, profiles,45-class mappings and zero-input inference; " + str(args.output))


if __name__ == "__main__":
    main()

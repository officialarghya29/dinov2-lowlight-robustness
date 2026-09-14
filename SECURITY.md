# Security Policy

## Supported versions

Only the `main` branch is supported.

## Dependency scanning

`.github/workflows/security.yml` runs [`pip-audit`](https://pypi.org/project/pip-audit/)
against `requirements.txt` on every change to it and weekly. Failures mean a
known CVE exists in the pinned-resolution of our declared dependencies.

## Reporting a vulnerability

Please do **not** report security issues in public issues.
Use GitHub's [private vulnerability reporting](https://github.com/officialarghya29/dinov2-lowlight-robustness/security/advisories/new)
so details stay confidential until a fix is released.

## Historical note

An exposed GitHub token was revoked by the owner in September 2026. It never
entered this repository's working tree or git history (verified by `git log -S`
and a full-tree scan), so no history rewrite was required.

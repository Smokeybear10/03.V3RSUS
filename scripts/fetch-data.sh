#!/bin/bash
# Pull the latest data artifacts from the private 802-DATA.MMA repo.
# Auth source order:
#   1. $GITHUB_TOKEN env var (Vercel + CI)
#   2. `gh auth token` (local dev with GitHub CLI authenticated)
#
# Run locally:  bash scripts/fetch-data.sh
# Vercel:       set as buildCommand in vercel.json (needs GITHUB_TOKEN env var)

set -euo pipefail

REPO="Smokeybear10/802-DATA.MMA"
TAG="${DATA_TAG:-latest}"
TARGET_DIR="${TARGET_DIR_OVERRIDE:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$TARGET_DIR"

# Resolve token
TOKEN="${GITHUB_TOKEN:-}"
if [ -z "$TOKEN" ] && command -v gh >/dev/null 2>&1; then
  TOKEN="$(gh auth token 2>/dev/null || true)"
fi

if [ -z "$TOKEN" ]; then
  echo "ERROR: need a GitHub token to fetch from private $REPO." >&2
  echo "Set GITHUB_TOKEN or run \`gh auth login\` first." >&2
  exit 1
fi

AUTH_HEADER="Authorization: Bearer $TOKEN"
API_BASE="https://api.github.com/repos/$REPO"

echo "Resolving release $TAG from $REPO..."
RELEASE_JSON="$(curl -fsSL -H "$AUTH_HEADER" -H "Accept: application/vnd.github+json" \
  "$API_BASE/releases/tags/$TAG")"

# Pipe JSON through python to extract id/name pairs without escaping headaches
ASSETS="$(printf '%s' "$RELEASE_JSON" | python3 -c 'import json,sys
for a in json.load(sys.stdin).get("assets", []):
    print(str(a["id"]) + "|" + a["name"])
')"

if [ -z "$ASSETS" ]; then
  echo "ERROR: release $TAG has no assets." >&2
  exit 1
fi

echo "$ASSETS" | while IFS='|' read -r ID NAME; do
  echo "  $NAME"
  curl -fsSL -H "$AUTH_HEADER" -H "Accept: application/octet-stream" \
    "$API_BASE/releases/assets/$ID" -o "$TARGET_DIR/$NAME"
done

echo ""
echo "Done. Files in $TARGET_DIR:"
ls -lh "$TARGET_DIR"

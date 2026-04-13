#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_VENV="${SOURCE_VENV:-$SCRIPT_DIR/.venv}"
TARGET_VENV="${TARGET_VENV:-$SCRIPT_DIR/.venv}"
LOCK_FILE="${LOCK_FILE:-$SCRIPT_DIR/requirements.venv.lock.txt}"
EDITABLES_FILE="${EDITABLES_FILE:-$SCRIPT_DIR/requirements.editable.local.txt}"
OPENPI_DIR="$SCRIPT_DIR/openpi"
OPENPI_CLIENT_DIR="$OPENPI_DIR/packages/openpi-client"

usage() {
  echo "Usage: $0 [snapshot|install|sync]"
  echo ""
  echo "  snapshot  Export pinned dependencies from SOURCE_VENV into LOCK_FILE."
  echo "  install   Create/use TARGET_VENV and install from LOCK_FILE + local editables."
  echo "  sync      snapshot (if SOURCE_VENV exists), then install."
  echo ""
  echo "Environment overrides:"
  echo "  SOURCE_VENV, TARGET_VENV, LOCK_FILE, EDITABLES_FILE"
}

ensure_editables_file() {
  : > "$EDITABLES_FILE"
  if [[ -d "$OPENPI_CLIENT_DIR" ]]; then
    echo "-e ./openpi/packages/openpi-client" >> "$EDITABLES_FILE"
  fi
  if [[ -d "$OPENPI_DIR" ]]; then
    echo "-e ./openpi" >> "$EDITABLES_FILE"
  fi
}

snapshot_from_venv() {
  if [[ ! -x "$SOURCE_VENV/bin/python" ]]; then
    echo "SOURCE_VENV not found: $SOURCE_VENV" >&2
    exit 1
  fi
  source "$SOURCE_VENV/bin/activate"
  local py_ver
  py_ver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
  {
    echo "# Generated from $SOURCE_VENV"
    echo "# Python $py_ver"
    echo "# Re-generate with: ./setup_environment.sh snapshot"
    python -m pip freeze --exclude-editable
  } > "$LOCK_FILE"
  ensure_editables_file
  echo "Wrote lock file: $LOCK_FILE"
  echo "Wrote local editable list: $EDITABLES_FILE"
}

install_from_lock() {
  if [[ ! -f "$LOCK_FILE" ]]; then
    echo "Lock file not found: $LOCK_FILE" >&2
    echo "Run '$0 snapshot' on the source machine first." >&2
    exit 1
  fi

  if [[ ! -x "$TARGET_VENV/bin/python" ]]; then
    python3 -m venv "$TARGET_VENV"
  fi
  source "$TARGET_VENV/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$LOCK_FILE"

  ensure_editables_file
  if [[ -s "$EDITABLES_FILE" ]]; then
    python -m pip install -r "$EDITABLES_FILE"
  fi

  python -c "import openpi; import openpi_client; print('Environment ready.')"
}

main() {
  local mode="${1:-sync}"
  case "$mode" in
    snapshot)
      snapshot_from_venv
      ;;
    install)
      install_from_lock
      ;;
    sync)
      if [[ -x "$SOURCE_VENV/bin/python" ]]; then
        snapshot_from_venv
      else
        echo "SOURCE_VENV missing, skipping snapshot and using existing lock."
      fi
      install_from_lock
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"

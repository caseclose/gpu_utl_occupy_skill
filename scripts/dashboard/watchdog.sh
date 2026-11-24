#!/usr/bin/env bash
set -eu
umask 077

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s STATE_DIR [collect options]\n' "$0"
    printf 'Checks the collector every 15 minutes. Use the same state directory and hardware options as agent.py collect.\n'
    exit 0
fi
if (( $# < 1 )); then
    printf 'A state directory is required. See %s --help.\n' "$0" >&2
    exit 2
fi
state=$1
shift
for arg in "$@"; do
    case "$arg" in
        --state|--state=*|--recover-monitor) echo "Specify the state directory once, as the first argument." >&2; exit 2 ;;
    esac
done
base=$(cd -- "$(dirname -- "$0")" && pwd)
python=${GPU_OCCUPY_PYTHON:-python3}
interval=900
for command in "$python" flock setsid; do
    command -v "$command" >/dev/null || { printf '%s is required\n' "$command" >&2; exit 1; }
done
mkdir -p -- "$state"
state=$(cd -- "$state" && pwd -P)
exec 8>"$state/watchdog.lock"
if ! flock -n 8; then
    printf 'A watchdog already holds the lock for %s\n' "$state" >&2
    exit 1
fi

wake_pid=
trap ':' USR1
trap 'exit 0' TERM INT
trap 'if [[ -n "$wake_pid" ]]; then kill "$wake_pid" 2>/dev/null || :; fi' EXIT

check_collector() {
    local status
    if flock -n -E 75 "$state/collector.lock" true; then
        nohup setsid "$python" "$base/agent.py" collect "$@" --state "$state" --recover-monitor \
            8>&- </dev/null >>"$state/collector.log" 2>&1 &
        printf '%s collector restart requested (pid %s)\n' "$(date -Is)" "$!"
    else
        status=$?
        if (( status != 75 )); then
            printf '%s collector lock check failed (exit %s)\n' "$(date -Is)" "$status" >&2
        fi
    fi
}

while true; do
    check_collector "$@"
    sleep "$interval" 8>&- &
    wake_pid=$!
    wait "$wake_pid" || :
    kill "$wake_pid" 2>/dev/null || :
    wait "$wake_pid" 2>/dev/null || :
    wake_pid=
done

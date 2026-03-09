#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
DIM='\033[2m'
RESET='\033[0m'

IMAGE_NAME="${IMAGE_NAME:-chord-recognition}"
TAG="${TAG:-latest}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-chord-recognition}"

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║     chord-recognition  ·  run        ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

if ! docker image inspect "${IMAGE_NAME}:${TAG}" &>/dev/null; then
    echo -e "${RED}${BOLD}✘ Image not found:${RESET} ${IMAGE_NAME}:${TAG}"
    echo -e "  Run ${BOLD}./build.sh${RESET} first."
    exit 1
fi

# Stop any existing container with the same name
if docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
    echo -e "${YELLOW}▶ Stopping existing container:${RESET} ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" >/dev/null
fi

echo -e "${YELLOW}▶ Starting container:${RESET} ${CONTAINER_NAME}"
echo

docker run --rm \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:7860" \
    -e PORT=7860 \
    "${IMAGE_NAME}:${TAG}" &

DOCKER_PID=$!

# Wait for the server to become ready
echo -ne "${YELLOW}  Waiting for server"
for i in $(seq 1 20); do
    if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
        break
    fi
    echo -n "."
    sleep 0.5
done
echo -e "${RESET}"

if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
    echo -e "${GREEN}${BOLD}✔ Server is up${RESET}"
    echo
    echo -e "  ${BOLD}Endpoints${RESET}"
    echo -e "  ${DIM}──────────────────────────────────────${RESET}"
    echo -e "  API    ${BOLD}http://localhost:${PORT}${RESET}"
    echo -e "  Docs   ${BOLD}http://localhost:${PORT}/docs${RESET}"
    echo -e "  Health ${BOLD}http://localhost:${PORT}/health${RESET}"
    echo
    echo -e "  ${BOLD}Open in browser${RESET}"
    echo -e "  ${DIM}──────────────────────────────────────${RESET}"
    echo -e "  ${CYAN}http://localhost:${PORT}/docs${RESET}"
    echo
    echo -e "  ${BOLD}Quick test with curl${RESET}"
    echo -e "  ${DIM}──────────────────────────────────────${RESET}"
    echo -e "  ${DIM}curl -X POST http://localhost:${PORT}/recognize \\${RESET}"
    echo -e "  ${DIM}  -F \"file=@audio.wav\"${RESET}"
    echo
    echo -e "${CYAN}  Press Ctrl+C to stop${RESET}"
    wait "${DOCKER_PID}"
else
    echo -e "${RED}${BOLD}✘ Server did not become ready${RESET}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    exit 1
fi

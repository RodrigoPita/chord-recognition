#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

IMAGE_NAME="${IMAGE_NAME:-chord-recognition}"
TAG="${TAG:-latest}"

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║     chord-recognition  ·  build      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "${YELLOW}▶ Building image:${RESET} ${IMAGE_NAME}:${TAG}"
echo

if docker build -t "${IMAGE_NAME}:${TAG}" .; then
    echo
    echo -e "${GREEN}${BOLD}✔ Build successful${RESET}"
    echo -e "  Image : ${BOLD}${IMAGE_NAME}:${TAG}${RESET}"
    echo -e "  Size  : $(docker image inspect "${IMAGE_NAME}:${TAG}" --format '{{.Size}}' | numfmt --to=iec)"
    echo
    echo -e "  Image is ready. Start the server with:"
    echo -e "  ${CYAN}${BOLD}./run.sh${RESET}"
else
    echo
    echo -e "${RED}${BOLD}✘ Build failed${RESET}"
    exit 1
fi

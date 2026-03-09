#!/usr/bin/env bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
DIM='\033[2m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║  chord-recognition  ·  deploy (HF)  ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${RESET}"

# --- Resolve username and space name ---
HF_USERNAME="${HF_USERNAME:-}"
SPACE_NAME="${SPACE_NAME:-chord-recognition}"

if [[ -z "${HF_USERNAME}" ]]; then
    echo -ne "${YELLOW}  Hugging Face username:${RESET} "
    read -r HF_USERNAME
fi

if [[ -z "${HF_USERNAME}" ]]; then
    echo -e "${RED}${BOLD}✘ Username is required.${RESET}"
    exit 1
fi

HF_REMOTE="git@hf.co:spaces/${HF_USERNAME}/${SPACE_NAME}"
HF_SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo
echo -e "  ${BOLD}Space${RESET}  : ${CYAN}${HF_SPACE_URL}${RESET}"
echo -e "  ${BOLD}Branch${RESET} : main"
echo

# --- Ensure hf.co is in known hosts ---
if ! ssh-keygen -F hf.co &>/dev/null; then
    echo -e "${YELLOW}▶ Adding hf.co to known hosts${RESET}"
    ssh-keyscan hf.co >> ~/.ssh/known_hosts 2>/dev/null
fi

# --- Check the remote exists or inform the user to create it ---
echo -e "${YELLOW}▶ Checking git remote${RESET}"

if ! git remote get-url hf &>/dev/null; then
    git remote add hf "${HF_REMOTE}"
    echo -e "  Added remote ${BOLD}hf${RESET} → ${HF_REMOTE}"
else
    EXISTING=$(git remote get-url hf)
    if [[ "${EXISTING}" != "${HF_REMOTE}" ]]; then
        git remote set-url hf "${HF_REMOTE}"
        echo -e "  Updated remote ${BOLD}hf${RESET} → ${HF_REMOTE}"
    else
        echo -e "  Remote ${BOLD}hf${RESET} already configured"
    fi
fi

echo
echo -e "${DIM}  Make sure you have created the Space first at:${RESET}"
echo -e "  ${CYAN}https://huggingface.co/new-space${RESET}"
echo -e "${DIM}  SDK → Docker  |  Space name → ${SPACE_NAME}${RESET}"
echo
echo -ne "${YELLOW}  Ready to push? [y/N] ${RESET}"
read -r CONFIRM

if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
    echo -e "${YELLOW}  Aborted.${RESET}"
    exit 0
fi

echo
echo -e "${YELLOW}▶ Pushing to Hugging Face Spaces${RESET}"
echo

git push hf main

echo
echo -e "${GREEN}${BOLD}✔ Pushed successfully${RESET}"
echo
echo -e "  ${BOLD}Your Space${RESET}"
echo -e "  ${DIM}──────────────────────────────────────${RESET}"
echo -e "  ${CYAN}${HF_SPACE_URL}${RESET}"
echo
echo -e "  ${DIM}HF will now build and deploy your Docker image.${RESET}"
echo -e "  ${DIM}This takes a few minutes on first deploy.${RESET}"
echo -e "  ${DIM}Once live, the API will be available at:${RESET}"
echo -e "  ${CYAN}https://${HF_USERNAME}-${SPACE_NAME}.hf.space${RESET}"
echo
echo -e "  ${BOLD}Docs${RESET}   ${CYAN}https://${HF_USERNAME}-${SPACE_NAME}.hf.space/docs${RESET}"
echo -e "  ${BOLD}Health${RESET} ${CYAN}https://${HF_USERNAME}-${SPACE_NAME}.hf.space/health${RESET}"

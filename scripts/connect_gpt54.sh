#!/usr/bin/env bash
# Configure the GPT-5.4 Azure OpenAI deployment for OpenAI-compatible clients.
# Source this file so the exported variables remain in the current shell:
#   source scripts/connect_gpt54.sh

export AZURE_OPENAI_ENDPOINT_54="http://127.0.0.1:8765/"
export AZURE_OPENAI_RESPONSES_PATH_54="/openai/responses?api-version=2025-04-01-preview"
export AZURE_OPENAI_DEPLOYMENT_54="gpt-5.4"
export AZURE_OPENAI_USE_AAD_54=1
export AZURE_OPENAI_AAD_SCOPE="https://cognitiveservices.azure.com/"

gpt54_refresh_token() {
  export AZURE_OPENAI_AAD_TOKEN_54="$(
    az account get-access-token \
      --resource "${AZURE_OPENAI_AAD_SCOPE}" \
      --query accessToken -o tsv
  )"

  # The local proxy injects AAD. Trace2Skill uses these for AzureResponsesClient.
  export OPENAI_API_KEY="EMPTY"
  export OPENAI_BASE_URL="${AZURE_OPENAI_ENDPOINT_54%/}/openai/"
  export OPENAI_MODEL="${AZURE_OPENAI_DEPLOYMENT_54}"
  export OPENAI_API_VERSION="2025-04-01-preview"
}

gpt54_refresh_token

echo "[INFO] GPT-5.4 configured through the local Azure Responses proxy."
echo "[INFO] Deployment: ${OPENAI_MODEL}"
echo "[INFO] Base URL: ${OPENAI_BASE_URL}"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[INFO] To keep these exports in your shell, run: source scripts/connect_gpt54.sh"
fi

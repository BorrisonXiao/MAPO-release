#!/usr/bin/env bash

resolve_answer_extraction_mode() {
    local default_mode="${1:-post_think}"
    shift || true

    local hint="${*,,}"
    if [[ "${hint}" == *"thinking"* ]]; then
        printf '%s\n' "post_think"
        return 0
    fi

    if [[ "${hint}" == *"instruct"* ]] || [[ "${hint}" == *"captioner"* ]]; then
        printf '%s\n' "answer_tag"
        return 0
    fi

    printf '%s\n' "${default_mode}"
}

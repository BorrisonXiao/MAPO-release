#!/usr/bin/env bash

# parse_options.sh
# Standard kaldi-style option parser.
# Usage: . utils/parse_options.sh
# Parses --key value arguments and assigns them to shell variables.

while true; do
    [ -z "${1:-}" ] && break
    case "$1" in
        --*=*)
            name=$(echo "$1" | sed 's/--\(.*\)=.*/\1/' | tr '-' '_')
            value=$(echo "$1" | sed 's/--[^=]*=//')
            eval "export ${name}=\"${value}\""
            shift 1
            ;;
        --*)
            name=$(echo "$1" | sed 's/--//' | tr '-' '_')
            value="$2"
            eval "export ${name}=\"${value}\""
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

#!/bin/bash

hf_download() {
    local repo=$1
    local local_dir=$2
    local use_uv=$3
    if [ -z "$local_dir" ]; then
        local_dir="$repo"
    fi

    echo "Downloading $repo to $local_dir..."

    local cmd=(hf download "$repo" --repo-type model --local-dir "$local_dir")

    if [ "$use_uv" = "--uv" ]; then
        cmd=(uv run "${cmd[@]}")
    fi

    HF_HUB_DISABLE_XET=False "${cmd[@]}"
}

hf_download "$1" "$2" "$3"

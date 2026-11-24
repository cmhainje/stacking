#!/bin/bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
BASE_URL="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"

mkdir -p "$DATA_DIR"

fetch() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.part"

    if [[ -f "$dest" ]]; then
        echo "already have $(basename "$dest"), skipping."
        return
    fi

    echo "downloading $(basename "$dest")..."
    curl -fL -C - -o "$tmp" "$url"
    mv "$tmp" "$dest"
}

for CAP in NGC SGC; do
    fetch "${BASE_URL}/LRG_${CAP}_clustering.dat.fits" \
        "${DATA_DIR}/LRG_${CAP}_clustering.dat.fits"
done

for CAP in NGC SGC; do
    for N in 0 1; do
        fetch "${BASE_URL}/LRG_${CAP}_${N}_clustering.ran.fits" \
            "${DATA_DIR}/LRG_${CAP}_${N}_clustering.ran.fits"
    done
done

for CAP in NGC SGC; do
    fetch "${BASE_URL}/ELG_LOPnotqso_${CAP}_clustering.dat.fits" \
        "${DATA_DIR}/ELG_LOPnotqso_${CAP}_clustering.dat.fits"
done

for CAP in NGC SGC; do
    for N in 0 1; do
        fetch "${BASE_URL}/ELG_LOPnotqso_${CAP}_${N}_clustering.ran.fits" \
            "${DATA_DIR}/ELG_LOPnotqso_${CAP}_${N}_clustering.ran.fits"
    done
done

echo "done."

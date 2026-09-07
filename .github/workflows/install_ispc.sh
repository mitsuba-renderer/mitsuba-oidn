#!/bin/bash
# Downloads a binary release of ISPC into ext/ispc (Linux and macOS)
set -euo pipefail

version=1.30.0
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  suffix=linux ;;
  Linux-aarch64) suffix=linux.aarch64 ;;
  Darwin-*)      suffix=macOS.universal ;;
  *) echo "unsupported platform"; exit 1 ;;
esac

cd "$(dirname "$0")/../.."
rm -rf ext/ispc
curl -sSL "https://github.com/ispc/ispc/releases/download/v${version}/ispc-v${version}-${suffix}.tar.gz" | tar xz -C ext
mv "ext/ispc-v${version}-${suffix}" ext/ispc
ext/ispc/bin/ispc --version

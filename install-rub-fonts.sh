#!/bin/bash

# We install the fonts locally into ~/.fonts/, not into /usr/share/fonts/truetype to avoid unnecessary sudo rights
DEST_DIR="${HOME}/.fonts/rub-flama"
mkdir -p $DEST_DIR

# Download six *.ttf files
VARIANTS='RubFlama-Bold RubFlama-BoldItalic RubFlama-Italic RubFlama-Regular RubFlamaLight-Italic RubFlamaLight-Regular'

for VARIANT in $VARIANTS; do
  # Do not overwrite existing files
  wget --no-clobber https://git.inf.bi.ruhr-uni-bochum.de/iib_pub/BeamerThemeIIB/-/raw/master/fonts/truetype/rub/rubflama/${VARIANT}.ttf -O "${DEST_DIR}/${VARIANT}.ttf"
done

# Update font cache
fc-cache -fv

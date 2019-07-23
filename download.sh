#!/bin/bash

set -e

DATADIR=$1

if [ ! -d "$DATADIR" ]; then
    mkdir -p "$DATADIR/original/play_by_play"
    kaggle competitions download -c mens-machine-learning-competition-2019 -p "$DATADIR/compressed"

    unzip "$DATADIR/compressed/Stage2DataFiles.zip" -d "$DATADIR/original/"
    unzip "$DATADIR/compressed/MasseyOrdinals.zip" -d "$DATADIR/original/"
    unzip "$DATADIR/compressed/PlayByPlay_2010.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2011.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2012.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2013.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2014.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2015.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2016.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2017.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2018.zip" -d "$DATADIR/original/play_by_play"
    unzip "$DATADIR/compressed/PlayByPlay_2019.zip" -d "$DATADIR/original/play_by_play"
fi

#!/bin/sh

mkdir -p data

wget -O data/kplr_dr25_inj1_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj1_plti.txt
wget -O data/kplr_dr25_inj2_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj2_plti.txt
wget -O data/kplr_dr25_inj3_plti.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj3_plti.txt

wget -O data/kplr_dr25_inj1_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj1_tces.txt
wget -O data/kplr_dr25_inj2_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj2_tces.txt
wget -O data/kplr_dr25_inj3_tces.txt https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/kplr_dr25_inj3_tces.txt

wget -O data/q1_q17_dr25_stellar.txt "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_stellar&format=ipac&select=*"
wget -O data/q1_q17_dr25_koi.txt "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_koi&format=ipac&select=*"

wget -O data/FLTI-DR25-KSOC-5006.tar.gz https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated/FLTI-DR25-KSOC-5006.tar.gz
cd data
tar -xf FLTI-DR25-KSOC-5006.tar.gz
cd ..

# This is too much data!
#wget -O data/wget_PDM.bat https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/wget_PDM.bat
#echo "downloading planet detection metrics..."
#cat data/wget_PDM.bat | grep fits | awk '{print $4}' | xargs -n 1 -P 24 wget --quiet -P data/pdm -x -nH -nc --cut-dirs=2

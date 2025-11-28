#!/bin/bash
pretrained="./param/GraphCast_operational_resolution_0.25_pressure_levels_13.npz"
dataset="./dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc"
mode="Checkpoint"
var="2m_temperature"

python inference.py --pretrained ${pretrained} --dataset ${dataset} --mode ${mode} --var ${var}

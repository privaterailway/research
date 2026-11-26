#!/bin/sh

echo "Cleaning the project artifacts"
rm -rf outputs results runs
rm -f model_scenario_?.pth
echo "DONE!"

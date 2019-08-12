#!/bin/bash
for file in $1
do
    echo $file
    pydeface $file
done  

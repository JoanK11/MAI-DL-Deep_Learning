#!/bin/bash

tar -czvf files.tar.gz $1
scp -r files.tar.gz nct01082@transfer1.bsc.es:~
rm files.tar.gz
ssh nct01082@alogin1.bsc.es << EOF
rm -r $1
tar -xzvf files.tar.gz
rm files.tar.gz
EOF
echo "Done!"
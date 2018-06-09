
STEPSIZE=50000
<<<<<<< HEAD
=======

>>>>>>> 87e120909317d4b9b6a4c998912dc039aeb5533f
mkdir -p output/samplesize

for SAMPLESIZE in 2 4 8 16 32 64 128 256
do
    for ITERATION in {2..10}
    do
        python trainer/task.py --train-files ./data/new/* --job-dir ./output/samplesize/$SAMPLESIZE --sample_size $SAMPLESIZE --train-steps $(($ITERATION * $STEPSIZE))
<<<<<<< HEAD
        python generate.py --samples 1000 --out_path ./output/samplesize/notalk\_$SAMPLESIZE\_$(($ITERATION * $STEPSIZE)) --dat_seed ./data/filtered/output.dat output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE))
        
=======
        python generate.py --samples 1000 --out_path ./output/samplesize/notalk\_$SAMPLESIZE\_$(($ITERATION * $STEPSIZE)) --dat_seed ./data/new/output.dat output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE))
>>>>>>> 87e120909317d4b9b6a4c998912dc039aeb5533f
        mkdir -p output/samplesize/backup/$SAMPLESIZE
        mkdir -p output/samplesize/$SAMPLESIZE
        cp output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE))* output/samplesize/backup/$SAMPLESIZE/
<<<<<<< HEAD
        exit
=======
    exit
>>>>>>> 87e120909317d4b9b6a4c998912dc039aeb5533f
    done
done


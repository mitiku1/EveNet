
STEPSIZE=50000
mkdir -p output/samplesize

for SAMPLESIZE in 2 4 8 16 32 64 128 256
do
    for ITERATION in {2..10}
    do
        # python trainer/task.py --train-files ./data/new/* --job-dir ./output/samplesize/$SAMPLESIZE --sample_size $SAMPLESIZE --train-steps $(($ITERATION * $STEPSIZE))
        python generate.py --samples 1000 --out_path ./output/samplesize/notalk\_$SAMPLESIZE\_1 output/samplesize/$SAMPLESIZE/model.ckpt-1
        
        mkdir -p output/samplesize/backup/$SAMPLESIZE
        mkdir -p output/samplesize/$SAMPLESIZE
        cp output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE))* output/samplesize/backup/$SAMPLESIZE/
        exit
    done
done


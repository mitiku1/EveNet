
STEPSIZE = 10  # 50000

mkdir output/samplesize

for SAMPLESIZE in 1 2 4 8 16 32 64 128 256
do
    for ITERATION in {1..10}
    do
        echo python trainer/task.py --train-files ./data/filtered/output.* \
                                    --job-dir ./output/samplesize/$SAMPLESIZE \
                                    --sample_size $SAMPLESIZE \
                                    --train-steps $(($ITERATION * $STEPSIZE))

        echo python generate.py --samples 1000 \
                                --out_path ./output/samplesize/notalk\_$SAMPLESIZE\_$(($ITERATION * $STEPSIZE)) 
                                --dat_seed ./data/filtered/output.dat 
                                output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE))

        cp output/samplesize/$SAMPLESIZE/model.ckpt-$(($ITERATION * $STEPSIZE)) \
           output/samplesize/backup/$SAMPLESIZE-model.ckpt-$(($ITERATION * 50000))
    done
done


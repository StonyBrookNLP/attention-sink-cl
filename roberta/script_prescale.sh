for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=3 python main.py --sequence news_series --lr 1e-5 --bsize 32 --epoch 5 --seed $seed >> logs/Prescale_news_series_${seed}.txt
done

for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=3 python main.py --sequence yahoo_split --lr 1e-5 --bsize 32 --epoch 5 --seed $seed >> logs/Prescale_news_series_${seed}.txt
done

for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=3 python main.py --sequence db_split --lr 1e-5 --bsize 32 --epoch 5 --seed $seed >> logs/Prescale_news_series_${seed}.txt
done

for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=3 python main.py --sequence news_series db_split yahoo_split --lr 1e-5 --bsize 32 --epoch 5 --seed $seed >> logs/Prescale_news_series_${seed}.txt
done

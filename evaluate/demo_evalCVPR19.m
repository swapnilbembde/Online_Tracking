diary('cvpr.txt');
disp('min_class_score = 0.1 and track score <0.3 max_time_lost = 35 all');
benchmarkGtDir = '/home/SharedData/swapnil/CVPR19/train/';
[allMets, metsBenchmark] = evaluateTracking('c11-train.txt', '../RESULT/cvpr_debug/', benchmarkGtDir, 'CVPR19');
diary('off');

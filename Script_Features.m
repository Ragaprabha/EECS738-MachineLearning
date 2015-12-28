% Script to plot MCC values for all 3 classifiers
RF = [0.61963 0.61991 0.61748 0.61773 0.61825 0.61733 0.61729 0.61802 0.6156 0.61724 0.61895 0.61771 0.61801 0.61665 0.61875 0.61711 0.61773 0.61712 0.61877 0.61911 0.61544];
AD = [0.9157 0.91581 0.91515 0.91552 0.91529 0.91566 0.91532 0.91518 0.91542 0.91554 0.91571 0.91550 0.91576 0.91580 0.91459 0.91541 0.91512 0.91551 0.91565 0.91542 0.91502];
%L = []

seed = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];

figure;
title('Seed vs MCC for classifiers');
xlabel('seed');
ylabel('MCC');
legend('y = RandomForest','y = ADTree')
hold on;
%plot(seed,RF,'g--o',seed,AD,'r--o',seed,L,'b--o');
plot(seed,RF,'g--o',seed,AD,'r--o');

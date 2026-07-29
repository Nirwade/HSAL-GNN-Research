
# Goodreads Young Adult (100k)

50 epochs, UB CCR.

## Config

batchSize 256, item/user_max_length 50, k_hop 2, lr 0.001, l2 0.0001, layer_num 3, hidden_size 50, 50 epochs.

## Results

Best-ever value per metric across the 50 epochs.

| K | Recall | NDCG | best epoch |

|---|---|---|---|

| 5 | 0.1452 | 0.0886 | 43 |

| 10 | 0.2459 | 0.1208 | 43 |

| 20 | 0.4102 | 0.1620 | 43 |

| 30 | 0.5329 | 0.1881 | 43 |

| 40 | 0.6287 | 0.2066 | 43 |

| 50 | 0.7080 | 0.2195 | 23 |

| 60 | 0.7773 | 0.2305 | 10 / 23 |

| 70 | 0.8427 | 0.2399 | 6 / 15 |

| 80 | 0.9008 | 0.2493 | 4 / 43 |

| 90 | 0.9488 | 0.2585 | 2 / 43 |

| 100 | 0.9963 | 0.2672 | 43 |

Final epoch (50) raw numbers: train_loss 0.1208, test_loss 10.5484.

---

# MAL (500k)

50 epochs, UB CCR.

## Config

batchSize 256, item/user_max_length 50, k_hop 2, lr 0.001, l2 0.0001, layer_num 3, hidden_size 50, 50 epochs.

## Results

Best-ever value per metric across the 50 epochs.

| K | Recall | NDCG | best epoch |

|---|---|---|---|

| 5 | 0.4587 | 0.2907 | 36 |

| 10 | 0.7202 | 0.3752 | 36 |

| 20 | 0.9270 | 0.4275 | 24 |

| 30 | 0.9771 | 0.4384 | 33 |

| 40 | 0.9909 | 0.4412 | 32 |

| 50 | 0.9965 | 0.4423 | 47 |

| 60 | 0.9988 | 0.4427 | 30 |

| 70 | 0.9997 | 0.4429 | 9 |

| 80 | 0.9999 | 0.4430 | 8 |

| 90 | 1.0000 | 0.4430 | 3 |

| 100 | 1.0000 | 0.4430 | 1 |

Final epoch (48) raw numbers: train_loss 0.1227, test_loss 7.2443.

---

MAL (500k) outperforms Goodreads young and adults 100k 


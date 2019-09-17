## DAGMM

```python
"""
2019-09-17 made by 스르륵
"""
```

- 나오게 된 배경 

```
  DAGMM jointly optimizes the parameters of the deep autoencoder and the mixture model simultaneously in an end-to-end fashion, leveraging a separate estimation network to facilitate the parameter learning of the mixture model.
```

  AutoEncoder + GMM을 한 번에 해결을 지향

```
  The joint optimization, which well bal- ances autoencoding reconstruction, density estimation of latent representation, and regularization, helps the autoencoder escape from less attractive local optima and further reduce reconstruction errors, avoiding the need of pre-training.
```

장점들.. 

1. less attractive local optima
2. reduce recon errors
3. avoid the need of retrain

![image-20190913201311307](/Users/Wook-Young/Library/Application Support/typora-user-images/image-20190913201311307.png)

- Deep AutoEncoder - Compression Network

AutoEncoder의 차원 축소 기능을 사용해서 Dimension Reduction 진행

Low dimension representation($Z_c$) 과 Reconstruction error Feature($Z_r$)로 다음 단계 학습

위 그림에서는 $Z_{c}$ concat with $Z_r$

$Z_c \sim  h(x; \theta_e),\ Z_r \sim f(x, x'), \ x' = g(Z_c; \theta_d) $

$Z = [Z_c, Z_r]$

$Z_r$ 은 다양한 거리 기준으로 바탕(Euclidean, relative Euclidean, cosine similarity)

- Estimation Network 

Estimation  Network는 GMM framework 아래에서 Density estimation을 진행

GMM의 $\phi, \mu, \Sigma $ 파라메터 추정에 사용 Multi Layer Network 사용

![image-20190913215328725](/Users/Wook-Young/Library/Application Support/typora-user-images/image-20190913215328725.png)

$\hat\gamma$ 은 MLN의 softmax 값인 K-dimension vector (membership prediction of for $z_i$)

![image-20190917104222861](/Users/Wook-Young/Library/Application Support/typora-user-images/image-20190917104222861.png)

Energy(Expectation of z) 는 위에와 같음.

- Objective Function

![image-20190917104342396](/Users/Wook-Young/Library/Application Support/typora-user-images/image-20190917104342396.png)

하나하나 분해해서 알아보자!

1. $L(x_i, x'_i)$ 

Loss function of Recon errors in compression network 

즉 AutoEncoder에서 나오는 로스 

논문에 의하면 주로 $L_2$가 성능이 좋다고 한다. 



2. $E(z_i)$ 는 앞서 설명한 Sample energy

```
  By minimizing the sample energy, we look for the best combination of compression and estimation networks that maximize the likelihood to observe input samples.
```

MLE 개념으로 가는듯



3. $P(\hat\Sigma)$ 

DAGMM 역시 Singularity 문제가 발생함. 이를 해결하기 위해 penalize 개념으로 집어 넣음

covariance matrix의 digonal entries의 sum 



4. $\lambda_1, \lambda_2$ 는 각각 0.1, 0.005로 둔다.



- Relation to Variational Inference

![image-20190917113840268](/Users/Wook-Young/Library/Application Support/typora-user-images/image-20190917113840268.png)

다음과 같은 Upper bound 식으로 (8)을 줄인다면! sample energy($E(z_i)$)를 줄이기 가능

```
  In DAGMM, we use Equation (6) as a part of the objective function instead of its upper bound in Equation (10) simply because the energy function of DAGMM is tractable and efficient to evaluate. Unlike neural variational inference that uses the deep estimation network to define a variational posterior distribution as described above, DAGMM explicitly employs the deep estimation network to parametrize a sample- dependent prior distribution.
```

(6)이 계산하기 쉬워서 Objective function으로 사용

DAGMM은 prior distribution의 parameter를 추출하는데 estimation network 사용



- 실험 결과 

https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf

본 논문 참조!

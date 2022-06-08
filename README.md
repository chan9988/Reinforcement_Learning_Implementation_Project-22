# Replication Project

## Part 1 - Sample MDP

### SARSA

Applying SARSA with Boltzmann Softmax to the problem described in paper's figure 1 could lead to non-single convergent
points. Run the following codes to replicate it:

```shell
python3 code/sample_mdp_sarsa.py
```
<table>
<tr>
        <td><img src="data/sample_mdp/SARSA/SARSA_sampleMDP_Boltzmann%20Softmax_smoothed.png" width="300px"></th>
        <td><img src="data/sample_mdp/SARSA/SARSA_sampleMDP_Mellowmax_smoothed.png" width="300px"></td>
    </tr>
</table>

### Generalized Value Iteration (GVI)

In the convergence theorem of GVI, the operator with non-expansion property is proved convergent. Boltzmann Softmax

```shell
python3 code/sample_mdp_gvi.py
```

<table>
<tr>
  <th>Update Vectors</th>
  <td><img src="data/sample_mdp/GVI/BoltzmannSoftmax/GIF.gif" width="300px"></th>
  <td><img src="data/sample_mdp/GVI/Mellowmax/GIF.gif" width="300px"></th>
</tr>
<tr>
  <th>Fixed Points</th>
  <td><img src="data/sample_mdp/GVI/BoltzmannSoftmax/Boltzmann%20Softmax_fixed_points.png" width="300px"></td>
  <td><img src="data/sample_mdp/GVI/Mellowmax/Mellowmax_fixed_points.png" width="300px"></td>
</tr>
</table>

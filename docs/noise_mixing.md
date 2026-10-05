# Controlled Noise Mixing

Signal power is estimated as:

$$
P_x = \frac{1}{N}\sum_{i=1}^{N} x_i^2
$$

For clean speech $s$ and noise $n$, the signal-to-noise ratio is:

$$
\mathrm{SNR}_{dB}
=
10 \log_{10}
\left(
\frac{P_s}{P_n}
\right)
$$

To control the SNR, the noise waveform is scaled:

$$
n_{\text{scaled}} = \alpha n
$$

Because signal power depends on squared amplitude:

$$
P_{n,\text{scaled}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\alpha n_i)^2
$$

Since:

$$
(\alpha n_i)^2 = \alpha^2 n_i^2
$$

we obtain:

$$
P_{n,\text{scaled}}
=
\alpha^2
\frac{1}{N}
\sum_{i=1}^{N}
n_i^2
$$

Therefore:

$$
P_{n,\text{scaled}}
=
\alpha^2 P_n
$$

At an SNR of $0$ dB:

$$
P_s = P_{n,\text{scaled}}
$$

so:

$$
\alpha^2 P_n = P_s
$$

and therefore:

$$
\alpha
=
\sqrt{
\frac{P_s}{P_n}
}
$$

For example, with:

$$
P_s = 0.0025
$$

and:

$$
P_n = 0.00000228
$$

the required scaling factor at $0$ dB is approximately:

$$
\alpha
=
\sqrt{
\frac{0.0025}{0.00000228}
}
\approx 33.1
$$

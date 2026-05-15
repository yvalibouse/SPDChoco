/*
 * spdc_kernel.c — Optimized SPDC spectral-amplitude quadrature (v4)
 *
 * v4 changes:  two methods for computing singles.
 *
 *   Method 0 — LG mode sum (original).
 *     Decomposes the undetected photon into Laguerre–Gauss radial
 *     modes and sums |⟨p|Φ⟩|² up to P_MAX.  Fast when P_MAX is
 *     small.  Convergence degrades for strongly mismatched focusing.
 *
 *   Method 1 — Double-z analytic trace (SPDCalc-style).
 *     Traces over the undetected photon analytically using the
 *     Hille–Hardy identity  Σ_p L_p(x) L_p(y) t^p = (1-t)^{-1}
 *     exp[-(x+y)t/(1-t)].  The transverse (r) integrals become
 *     Gaussian and are performed in closed form, leaving a 2-D
 *     (z₁, z₂) quadrature.  No P_MAX truncation — the mode sum
 *     is exact.  Cost scales as N_Z², independent of the number
 *     of spatial modes.
 *
 *     Physics:  for signal singles, the idler is "bucket-detected"
 *     (all transverse modes collected).  The Hille–Hardy kernel
 *     replaces the explicit LG expansion of the bucket projector.
 *
 * Build (Linux/macOS):
 *   gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC \
 *       spdc_kernel.c -o libspdc.so -lm
 *
 * Build (Windows / MinGW):
 *   gcc -O3 -march=haswell -ffast-math -fopenmp -shared \
 *       spdc_kernel.c -o libspdc.dll
 */

#include <math.h>
#include <string.h>

static inline void sincos_portable(double x, double *s, double *c)
{ *s = sin(x); *c = cos(x); }
#define sincos sincos_portable

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#define MAX_P    40
#define MAX_NR   128
#define MAX_NZ   256

/* ── Beam helpers ──────────────────────────────────────────────── */

static inline double beam_w(double z, double k, double w0)
{
    double zR = 0.5 * k * w0 * w0;
    return w0 * sqrt(1.0 + (z / zR) * (z / zR));
}

static inline double inv_R_fn(double z, double k, double w0)
{
    double zR = 0.5 * k * w0 * w0;
    return z / (z * z + zR * zR);
}

static inline double gouy_fn(double z, double k, double w0)
{
    double zR = 0.5 * k * w0 * w0;
    return atan2(z, zR);
}

static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline int    imax(int a, int b)       { return a > b ? a : b; }

static int adaptive_pmax(double wp, double ws, double wi, double L,
                         double lam0_p, double lam0_s, double lam0_i,
                         double n0_p, double n0_s, double n0_i,
                         int P_MAX_MIN)
{
    double xi_p = L * lam0_p / (2.0 * M_PI * n0_p * wp * wp);
    double xi_s = L * lam0_s / (2.0 * M_PI * n0_s * ws * ws);
    double xi_i = L * lam0_i / (2.0 * M_PI * n0_i * wi * wi);
    double xi = dmax(xi_p, dmax(xi_s, xi_i));
    return imax(P_MAX_MIN, (int)ceil(1.5 * xi) + 2);
}


/* ── Vectorised Laguerre advance ──────────────────────────────── */

static inline void laguerre_advance(
    int nr, int p,
    const double * restrict x,
    const double * restrict Lprev,
    const double * restrict Lcurr,
    double       * restrict Lnext)
{
    double inv_p = 1.0 / (double)p;
    double c1 = (2.0 * p - 1.0) * inv_p;
    double c2 = inv_p;
    double c3 = (double)(p - 1) * inv_p;
    for (int ir = 0; ir < nr; ir++)
        Lnext[ir] = (c1 - c2 * x[ir]) * Lcurr[ir] - c3 * Lprev[ir];
}


/* ── Complex arithmetic helpers ────────────────────────────────── */

static inline void cdiv(double ar, double ai,
                        double br, double bi,
                        double *qr, double *qi)
{
    double d = br*br + bi*bi;
    *qr = (ar*br + ai*bi) / d;
    *qi = (ai*br - ar*bi) / d;
}

static inline void cmul(double ar, double ai,
                        double br, double bi,
                        double *pr, double *pi)
{
    *pr = ar*br - ai*bi;
    *pi = ar*bi + ai*br;
}


/* ══════════════════════════════════════════════════════════════════
 *  METHOD 0:  LG mode sum  (original kernel)
 * ══════════════════════════════════════════════════════════════════ */

static void one_spectral_point_lgsum(
    double k_s, double k_i, double dk, int P_MAX,
    double wp, double ws, double wi,
    double k_p, double k0_p, double L, double offset_w0,
    double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    int N_R, const double *r_ref,   const double *r_weights,
    double * restrict P_0i, double * restrict P_i0)
{
    int pm = P_MAX < MAX_P ? P_MAX : MAX_P;
    int nr = N_R < MAX_NR ? N_R : MAX_NR;

    double half_L = 0.5 * L;
    /* Per-beam wavelength-dependent waist offsets (Eq. 47 in the theory) */
    double off_p = offset_w0 + half_L * (k_p / k0_p - 1.0);
    double off_s = offset_w0 + half_L * (k_s / k0_s - 1.0);
    double off_i = offset_w0 + half_L * (k_i / k0_i - 1.0);
    double zp_max = dmax(fabs(half_L + off_p), fabs(half_L - off_p));
    double zs_max = dmax(fabs(half_L + off_s), fabs(half_L - off_s));
    double zi_max = dmax(fabs(half_L + off_i), fabs(half_L - off_i));
    double mode_scale = sqrt(2.0 * pm + 1.0);
    double r_cut = 4.0 * dmax(beam_w(zp_max, k_p, wp),
                        dmax(beam_w(zs_max, k_s, ws) * mode_scale,
                             beam_w(zi_max, k_i, wi) * mode_scale));
    double half_r = 0.5 * r_cut;
    double mid_r  = 0.5 * r_cut;

    double a0r[MAX_P+1], a0im[MAX_P+1];
    double air[MAX_P+1], aiim[MAX_P+1];
    memset(a0r,  0, (pm+1)*sizeof(double));
    memset(a0im, 0, (pm+1)*sizeof(double));
    memset(air,  0, (pm+1)*sizeof(double));
    memset(aiim, 0, (pm+1)*sizeof(double));

    double c_re[MAX_NR], c_im[MAX_NR];
    double arg_s[MAX_NR], arg_i[MAX_NR];
    double buf_a[MAX_NR], buf_b[MAX_NR], buf_c[MAX_NR];

    for (int iz = 0; iz < N_Z; iz++) {
        double z     = z_nodes[iz];
        double wz    = z_weights[iz] * chi_z[iz];   /* χ(z) factor */
        double z_p   = z + off_p;
        double z_s   = z + off_s;
        double z_i   = z + off_i;

        double w_p = beam_w(z_p, k_p,  wp);
        double w_s = beam_w(z_s, k_s,  ws);
        double w_i = beam_w(z_i, k_i,  wi);

        double inv_wp2 = 1.0/(w_p*w_p);
        double inv_ws2 = 1.0/(w_s*w_s);
        double inv_wi2 = 1.0/(w_i*w_i);
        double sum_inv  = inv_wp2 + inv_ws2 + inv_wi2;
        double norm     = 1.0/(w_p * w_s * w_i);

        double g_p = gouy_fn(z_p, k_p,  wp);
        double g_s = gouy_fn(z_s, k_s,  ws);
        double g_i = gouy_fn(z_i, k_i,  wi);

        double base_phase = z*dk + (-g_p + g_s + g_i);
        double curv = 0.5*k_p *inv_R_fn(z_p, k_p,  wp)
                    - 0.5*k_s *inv_R_fn(z_s, k_s,  ws)
                    - 0.5*k_i *inv_R_fn(z_i, k_i,  wi);

        double gs_re, gs_im, gi_re, gi_im;
        sincos(2.0*g_s, &gs_im, &gs_re);
        sincos(2.0*g_i, &gi_im, &gi_re);

        for (int ir = 0; ir < nr; ir++) {
            double r  = half_r * r_ref[ir] + mid_r;
            double wr = half_r * r_weights[ir] * r;
            double r2 = r * r;

            arg_s[ir] = 2.0 * r2 * inv_ws2;
            arg_i[ir] = 2.0 * r2 * inv_wi2;

            double amp   = norm * exp(-r2 * sum_inv);
            double phase = base_phase + r2 * curv;
            double sn, cs;
            sincos(phase, &sn, &cs);
            double w = wz * wr * amp;
            c_re[ir] = w * cs;
            c_im[ir] = w * sn;
        }

        /* Modes (0, pi): Laguerre in ξ_i, Gouy in ψ_i */
        double *Lprev_i = buf_a, *Lcurr_i = buf_b, *Lwork_i = buf_c;
        for (int ir = 0; ir < nr; ir++) Lprev_i[ir] = 1.0;
        double gip_re = 1.0, gip_im = 0.0;

        for (int pi = 0; pi <= pm; pi++) {
            const double *Lp;
            if (pi == 0) {
                Lp = Lprev_i;
            } else if (pi == 1) {
                for (int ir = 0; ir < nr; ir++)
                    Lcurr_i[ir] = 1.0 - arg_i[ir];
                Lp = Lcurr_i;
            } else {
                laguerre_advance(nr, pi, arg_i, Lprev_i, Lcurr_i, Lwork_i);
                double *tmp = Lprev_i;
                Lprev_i = Lcurr_i; Lcurr_i = Lwork_i; Lwork_i = tmp;
                Lp = Lcurr_i;
            }

            double dr = 0.0, di = 0.0;
            for (int ir = 0; ir < nr; ir++) {
                dr += c_re[ir] * Lp[ir];
                di += c_im[ir] * Lp[ir];
            }

            a0r[pi]  += dr * gip_re - di * gip_im;
            a0im[pi] += dr * gip_im + di * gip_re;

            double t = gip_re * gi_re - gip_im * gi_im;
            gip_im   = gip_re * gi_im + gip_im * gi_re;
            gip_re   = t;
        }

        /* Modes (ps, 0): Laguerre in ξ_s, Gouy in ψ_s */
        double *Lprev_s = buf_a, *Lcurr_s = buf_b, *Lwork_s = buf_c;
        for (int ir = 0; ir < nr; ir++) Lprev_s[ir] = 1.0;
        double gsp_re = 1.0, gsp_im = 0.0;

        for (int ps = 0; ps <= pm; ps++) {
            const double *Lp;
            if (ps == 0) {
                Lp = Lprev_s;
            } else if (ps == 1) {
                for (int ir = 0; ir < nr; ir++)
                    Lcurr_s[ir] = 1.0 - arg_s[ir];
                Lp = Lcurr_s;
            } else {
                laguerre_advance(nr, ps, arg_s, Lprev_s, Lcurr_s, Lwork_s);
                double *tmp = Lprev_s;
                Lprev_s = Lcurr_s; Lcurr_s = Lwork_s; Lwork_s = tmp;
                Lp = Lcurr_s;
            }

            if (ps > 0) {
                double t = gsp_re * gs_re - gsp_im * gs_im;
                gsp_im   = gsp_re * gs_im + gsp_im * gs_re;
                gsp_re   = t;

                double dr = 0.0, di = 0.0;
                for (int ir = 0; ir < nr; ir++) {
                    dr += c_re[ir] * Lp[ir];
                    di += c_im[ir] * Lp[ir];
                }
                air[ps]  += dr * gsp_re - di * gsp_im;
                aiim[ps] += dr * gsp_im + di * gsp_re;
            }
        }
    }

    double twopi = 2.0 * M_PI;
    for (int i = 0; i <= pm; i++) {
        double re = twopi * a0r[i], im = twopi * a0im[i];
        P_0i[i] = re*re + im*im;
    }
    P_i0[0] = P_0i[0];
    for (int i = 1; i <= pm; i++) {
        double re = twopi * air[i], im = twopi * aiim[i];
        P_i0[i] = re*re + im*im;
    }
}


/* ══════════════════════════════════════════════════════════════════
 *  METHOD 1:  Double-z analytic trace  (SPDCalc-style)
 *
 *  Coincidences S2:  single-z quadrature (p = 0 only), same as
 *                    method 0 but with P_MAX = 0.
 *
 *  Singles S1_s/i:   trace out idler/signal using Hille–Hardy.
 *
 *  After performing the transverse integral analytically:
 *
 *    ∫₀^∞ r exp(−σ r²) dr  =  1 / (2σ)
 *
 *  the double-z integrand becomes (for S1_s, tracing over idler):
 *
 *    I(z₁, z₂) = norm₁ × norm₂ × exp(i ΔΦ)
 *                 ─────────────────────────────
 *                     (1 − t) × 4 σ₁ σ₂
 *
 *  t    = exp( 2 i Δψ_i ),   Δψ_i = ψ_i(z₁) − ψ_i(z₂)
 *
 *  σ₁ = 1/w_p₁² + 1/w_s₁² + (1+t)/[(1−t) w_i₁²] − i curv₁
 *  σ₂ = 1/w_p₂² + 1/w_s₂² + conj[(1+t)/((1−t) w_i₂²)] + i curv₂
 *
 *  Diagonal z₁ = z₂:  t → 1,  (1+t)/(1−t) ~ i/Δψ → ∞,
 *  so σ ~ 1/(1−t), and  1/[(1−t) σ₁ σ₂] ~ (1−t) → 0.
 *  The integrand is regular at z₁ = z₂ (zero measure).
 *  We skip iz₁ == iz₂ in the quadrature.
 * ══════════════════════════════════════════════════════════════════ */

static void one_spectral_point_doublez(
    double k_s, double k_i, double dk,
    double wp, double ws, double wi,
    double k_p, double k0_p, double L, double offset_w0,
    double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    int N_R, const double *r_ref,   const double *r_weights,
    double *S2_out, double *S1_s_out, double *S1_i_out)
{
    int nz = N_Z < MAX_NZ ? N_Z : MAX_NZ;
    (void)N_R; (void)r_ref; (void)r_weights; /* unused in method 1 */

    double half_L = 0.5 * L;
    /* Per-beam wavelength-dependent waist offsets (Eq. 47) */
    double off_p = offset_w0 + half_L * (k_p / k0_p - 1.0);
    double off_s = offset_w0 + half_L * (k_s / k0_s - 1.0);
    double off_i = offset_w0 + half_L * (k_i / k0_i - 1.0);

    /* ── Step 1: S2 via ANALYTICAL transverse overlap ────────
     *
     *  I₀(z) = N(z) · exp(i·Φ(z)) / (2 A(z))
     *
     *  with N = 1/(w_p w_s w_i),
     *       Φ = −ψ_p + ψ_s + ψ_i         (Gouy phases),
     *       A = Σ 1/w² + i·(curvature)     (complex Gaussian width).
     *
     *  C₀₀ = 2π ∫ dz exp(i Δk z) I₀(z),  S₂ = |C₀₀|².
     *
     *  This replaces the numerical r-integral with the exact
     *  closed-form ∫₀^∞ r exp(−A r²) dr = 1/(2A).
     * ────────────────────────────────────────────────────────── */

    double acc_re = 0.0, acc_im = 0.0;

    for (int iz = 0; iz < nz; iz++) {
        double z     = z_nodes[iz];
        double wz    = z_weights[iz] * chi_z[iz];   /* χ(z) */
        double z_p   = z + off_p;
        double z_s   = z + off_s;
        double z_i   = z + off_i;

        double w_p = beam_w(z_p, k_p,  wp);
        double w_s = beam_w(z_s, k_s,  ws);
        double w_i = beam_w(z_i, k_i,  wi);

        double Re_A = 1.0/(w_p*w_p) + 1.0/(w_s*w_s) + 1.0/(w_i*w_i);
        double Im_A = 0.5*k_p *inv_R_fn(z_p, k_p,  wp)
                    - 0.5*k_s *inv_R_fn(z_s, k_s,  ws)
                    - 0.5*k_i *inv_R_fn(z_i, k_i,  wi);

        double norm_val = 1.0 / (w_p * w_s * w_i);

        double denom = 2.0 * (Re_A * Re_A + Im_A * Im_A);
        double inv2A_re = Re_A  / denom;
        double inv2A_im = Im_A  / denom;

        double phase = z * dk
            + (-gouy_fn(z_p, k_p, wp) + gouy_fn(z_s, k_s, ws) + gouy_fn(z_i, k_i, wi));

        double sn, cs;
        sincos(phase, &sn, &cs);

        double I_re = norm_val * (cs * inv2A_re - sn * inv2A_im);
        double I_im = norm_val * (sn * inv2A_re + cs * inv2A_im);

        acc_re += wz * I_re;
        acc_im += wz * I_im;
    }

    {
        double twopi = 2.0 * M_PI;
        double re = twopi * acc_re;
        double im = twopi * acc_im;
        *S2_out = re*re + im*im;
    }


    /* ── Step 2: precompute beam parameters at each z-node ──── */

    double b_wp[MAX_NZ],  b_ws[MAX_NZ],  b_wi[MAX_NZ];
    double b_curv[MAX_NZ], b_base[MAX_NZ];
    double b_gi[MAX_NZ],   b_gs[MAX_NZ];

    for (int iz = 0; iz < nz; iz++) {
        double z   = z_nodes[iz];
        double z_p = z + off_p;
        double z_s = z + off_s;
        double z_i = z + off_i;

        b_wp[iz] = beam_w(z_p, k_p,  wp);
        b_ws[iz] = beam_w(z_s, k_s,  ws);
        b_wi[iz] = beam_w(z_i, k_i,  wi);

        double gp = gouy_fn(z_p, k_p,  wp);
        b_gs[iz]  = gouy_fn(z_s, k_s,  ws);
        b_gi[iz]  = gouy_fn(z_i, k_i,  wi);

        b_curv[iz] = 0.5*k_p *inv_R_fn(z_p, k_p,  wp)
                   - 0.5*k_s *inv_R_fn(z_s, k_s,  ws)
                   - 0.5*k_i *inv_R_fn(z_i, k_i,  wi);

        b_base[iz] = z * dk + (-gp + b_gs[iz] + b_gi[iz]);
    }


    /* ── Step 3: double-z integrals (off-diagonal) ────────────── *
     *
     * The correct kernel for
     *   Σ_p R_p(z₁) R_p*(z₂) s^p
     * where R_p(z) = N(z)(w_t²/4)(α−1)^p / α^{p+1}  is the
     * COMPLETED radial integral at each z (single variable, not two),
     * is the geometric series:
     *
     *   F = N₁ N₂ w_{t1}² w_{t2}² / [16 · D]
     *
     * where D = α₁ α₂* − s (α₁−1)(α₂*−1)
     *         = α₁ α₂* (1−s) + s(α₁ + α₂* − 1)
     *
     *   α_j  = (σ_j − i C_j) w_{t,j}² / 2
     *   σ_j  = 1/w_p² + 1/w_s² + 1/w_i²     (all three beams)
     *   C_j  = ½ k_p/R_p − ½ k_s/R_s − ½ k_i/R_i
     *   s    = exp(2 i Δg_traced)
     *   w_t  = beam width of the traced-out arm
     *
     * For signal singles: t = idler,  s = exp(2i(g_{i1}−g_{i2}))
     * For idler  singles: t = signal, s = exp(2i(g_{s1}−g_{s2}))
     *
     * On the diagonal (z₁ = z₂): s = 1, α₁ = α₂ = α,
     *   D = |α|² − |α−1|² = 2 Re(α) − 1 → finite (Parseval).
     * This is handled separately in Step 4.
     * ─────────────────────────────────────────────────────────── */

    double S1_s_acc = 0.0;
    double S1_i_acc = 0.0;

    for (int iz1 = 0; iz1 < nz; iz1++) {
        double wz1 = z_weights[iz1] * chi_z[iz1];   /* χ(z₁) */

        double wp1_2 = b_wp[iz1] * b_wp[iz1];
        double ws1_2 = b_ws[iz1] * b_ws[iz1];
        double wi1_2 = b_wi[iz1] * b_wi[iz1];

        double sig1  = 1.0/wp1_2 + 1.0/ws1_2 + 1.0/wi1_2;
        double N1    = 1.0 / (b_wp[iz1] * b_ws[iz1] * b_wi[iz1]);

        /* α₁ = (σ₁ − iC₁)·w_i₁²/2  for signal singles (trace idler) */
        double a1i_re =  sig1 * wi1_2 * 0.5;
        double a1i_im = -b_curv[iz1] * wi1_2 * 0.5;

        /* β₁ = (σ₁ − iC₁)·w_s₁²/2  for idler singles (trace signal) */
        double b1s_re =  sig1 * ws1_2 * 0.5;
        double b1s_im = -b_curv[iz1] * ws1_2 * 0.5;

        for (int iz2 = 0; iz2 < nz; iz2++) {
            if (iz2 == iz1) continue;

            double wz2 = z_weights[iz2] * chi_z[iz2];   /* χ(z₂) */

            double wp2_2 = b_wp[iz2] * b_wp[iz2];
            double ws2_2 = b_ws[iz2] * b_ws[iz2];
            double wi2_2 = b_wi[iz2] * b_wi[iz2];

            double sig2  = 1.0/wp2_2 + 1.0/ws2_2 + 1.0/wi2_2;
            double N2    = 1.0 / (b_wp[iz2] * b_ws[iz2] * b_wi[iz2]);

            /* Full phase difference */
            double dphi = b_base[iz1] - b_base[iz2];
            double ep_re, ep_im;
            sincos(dphi, &ep_im, &ep_re);

            /* ── S1_s: trace over idler ─────────────────────── */
            {
                double dg = b_gi[iz1] - b_gi[iz2];
                double s_re, s_im;
                sincos(2.0 * dg, &s_im, &s_re);   /* s = exp(2iΔg_i) */

                /* α₂* = (σ₂ + iC₂)·w_i₂²/2 */
                double a2c_re =  sig2 * wi2_2 * 0.5;
                double a2c_im =  b_curv[iz2] * wi2_2 * 0.5;

                /* D = α₁ α₂* − s (α₁−1)(α₂*−1) */

                /* α₁ α₂* */
                double aa_re, aa_im;
                cmul(a1i_re, a1i_im, a2c_re, a2c_im, &aa_re, &aa_im);

                /* (α₁−1)(α₂*−1) */
                double a1m1_re = a1i_re - 1.0, a1m1_im = a1i_im;
                double a2m1_re = a2c_re - 1.0, a2m1_im = a2c_im;
                double mm_re, mm_im;
                cmul(a1m1_re, a1m1_im, a2m1_re, a2m1_im, &mm_re, &mm_im);

                /* s · (α₁−1)(α₂*−1) */
                double smm_re, smm_im;
                cmul(s_re, s_im, mm_re, mm_im, &smm_re, &smm_im);

                /* D = α₁α₂* − s(α₁−1)(α₂*−1) */
                double D_re = aa_re - smm_re;
                double D_im = aa_im - smm_im;

                /* F = N₁N₂ w_{i1}² w_{i2}² / (16 D) */
                double num = N1 * N2 * wi1_2 * wi2_2 / 16.0;
                double frac_re, frac_im;
                cdiv(num, 0.0, D_re, D_im, &frac_re, &frac_im);

                /* K(z₁,z₂) = F · exp(iΔΨ) */
                double I_re, I_im;
                cmul(frac_re, frac_im, ep_re, ep_im, &I_re, &I_im);

                S1_s_acc += wz1 * wz2 * I_re;
            }

            /* ── S1_i: trace over signal ────────────────────── */
            {
                double dg = b_gs[iz1] - b_gs[iz2];
                double s_re, s_im;
                sincos(2.0 * dg, &s_im, &s_re);   /* s = exp(2iΔg_s) */

                /* β₂* = (σ₂ + iC₂)·w_s₂²/2 */
                double b2c_re =  sig2 * ws2_2 * 0.5;
                double b2c_im =  b_curv[iz2] * ws2_2 * 0.5;

                /* D = β₁ β₂* − s (β₁−1)(β₂*−1) */
                double bb_re, bb_im;
                cmul(b1s_re, b1s_im, b2c_re, b2c_im, &bb_re, &bb_im);

                double b1m1_re = b1s_re - 1.0, b1m1_im = b1s_im;
                double b2m1_re = b2c_re - 1.0, b2m1_im = b2c_im;
                double mm_re, mm_im;
                cmul(b1m1_re, b1m1_im, b2m1_re, b2m1_im, &mm_re, &mm_im);

                double smm_re, smm_im;
                cmul(s_re, s_im, mm_re, mm_im, &smm_re, &smm_im);

                double D_re = bb_re - smm_re;
                double D_im = bb_im - smm_im;

                double num = N1 * N2 * ws1_2 * ws2_2 / 16.0;
                double frac_re, frac_im;
                cdiv(num, 0.0, D_re, D_im, &frac_re, &frac_im);

                double I_re, I_im;
                cmul(frac_re, frac_im, ep_re, ep_im, &I_re, &I_im);

                S1_i_acc += wz1 * wz2 * I_re;
            }
        }
    }


    /* ── Step 4: diagonal contribution (Parseval) ────────────── *
     *
     * The Mehler generating function diverges at z₁ = z₂ (s = 1),
     * but the physical mode sum  Σ_p |R_p(z)|²  converges.
     * We evaluate it analytically via Parseval's theorem on the
     * Laguerre basis.
     *
     *   R_p(z) = N(z) ∫₀^∞ L_p(ξ) exp(−αξ) · (w_traced²/4) dξ
     *          = N · (w_t²/4) · (α−1)^p / α^{p+1}
     *
     *   where ξ = 2r²/w_t², α = Γ₀·w_t²/2, Γ₀ = σ_total − iC.
     *
     *   Σ_p |R_p|² = N² · w_t⁴/16 · 1/(2Re(α)−1)
     *
     *   Now  2Re(α)−1 = Re(Γ₀)·w_t² − 1 = σ_total·w_t² − 1
     *                  = w_t²·σ_kept
     *
     *   where σ_kept = 1/w_p² + 1/w_kept²  (the two beams NOT traced).
     *
     * The curvature C drops out entirely: the total power into all
     * transverse modes depends only on the Gaussian overlap widths.
     *
     * Final formula:
     *   D_s(z) = N² · w_i² / (16 · σ_ps)    [trace over idler]
     *          = 1 / (16 · σ_ps · w_p² · w_s²)
     *
     *   D_i(z) = N² · w_s² / (16 · σ_pi)    [trace over signal]
     *          = 1 / (16 · σ_pi · w_p² · w_i²)
     * ─────────────────────────────────────────────────────────── */

    double S1_s_diag = 0.0;
    double S1_i_diag = 0.0;

    for (int iz = 0; iz < nz; iz++) {
        double chi2 = chi_z[iz] * chi_z[iz];            /* χ(z)² */
        double wz2 = z_weights[iz] * z_weights[iz] * chi2;
        double wp2 = b_wp[iz] * b_wp[iz];
        double ws2 = b_ws[iz] * b_ws[iz];
        double wi2 = b_wi[iz] * b_wi[iz];

        double sig_ps = 1.0/wp2 + 1.0/ws2;
        double sig_pi = 1.0/wp2 + 1.0/wi2;

        /* D_s = 1/(16 σ_ps w_p² w_s²) */
        S1_s_diag += wz2 / (16.0 * sig_ps * wp2 * ws2);

        /* D_i = 1/(16 σ_pi w_p² w_i²) */
        S1_i_diag += wz2 / (16.0 * sig_pi * wp2 * wi2);
    }


    /* ── Step 5: combine ─────────────────────────────────────── */

    double fourpi2 = 4.0 * M_PI * M_PI;
    *S1_s_out = fourpi2 * (S1_s_acc + S1_s_diag);
    *S1_i_out = fourpi2 * (S1_i_acc + S1_i_diag);
}


/* ══════════════════════════════════════════════════════════════════
 *  BATCHED METHOD 1:  All spectral points for one waist pair
 *
 *  Key optimisations over calling one_spectral_point_doublez() N_SPEC
 *  times in a loop:
 *
 *  1. Pump beam parameters (w_p, ψ_p, k_p/(2R_p)) are computed
 *     ONCE per z-node and reused across all spectral points.
 *
 *  2. S₂ (coincidences) is batched: a single z-loop accumulates
 *     N_SPEC complex amplitudes in parallel, avoiding N_SPEC
 *     independent z-loops.  Cost drops from O(N_SPEC × N_Z)
 *     to O(N_SPEC + N_Z × N_SPEC) = O(N_Z × N_SPEC) with a
 *     much smaller prefactor due to reusing pump params.
 *
 *  3. Singles: per-spectral-point double-z still dominates at
 *     O(N_Z²) per point, but pump params are reused and
 *     function-call overhead is eliminated.
 *
 *  Memory: O(N_SPEC) for S₂ accumulators  (heap-allocated),
 *          O(N_Z)    for pump tables       (stack).
 * ══════════════════════════════════════════════════════════════════ */

#include <stdlib.h>   /* malloc / free */

static void batch_doublez_one_waist(
    int N_SPEC,
    const double *k_s_arr, const double *k_i_arr,
    const double *dk_arr, const double *w_spec,
    const double *alpha_sq,
    const double *k_p_arr,
    double wp, double ws, double wi,
    double k0_p, double L, double offset_w0,
    double k0_s, double k0_i,
    int nz, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    double *S2_out, double *S1_s_out, double *S1_i_out)
{
    /* ── S₂: batched analytical z-integral ───────────────────
     *
     *  Accumulate C₀₀[il] = 2π Σ_z wz · I₀(z; k_s[il], k_i[il])
     *  for all spectral points simultaneously in one z-pass.
     *
     *  Pump beam params now vary per spectral point (broadband),
     *  so they cannot be precomputed across spectral points.
     * ────────────────────────────────────────────────────────── */

    double half_L = 0.5 * L;

    double *c_re = (double *)calloc(2 * (size_t)N_SPEC, sizeof(double));
    double *c_im = c_re + N_SPEC;

    for (int iz = 0; iz < nz; iz++) {
        double z     = z_nodes[iz];
        double wz    = z_weights[iz] * chi_z[iz];   /* χ(z) */

        for (int il = 0; il < N_SPEC; il++) {
            double k_s = k_s_arr[il];
            double k_i = k_i_arr[il];
            double k_p = k_p_arr[il];
            double dk  = dk_arr[il];

            /* Per-beam wavelength-dependent waist offsets (Eq. 47) */
            double z_p = z + offset_w0 + half_L * (k_p / k0_p - 1.0);
            double z_s = z + offset_w0 + half_L * (k_s / k0_s - 1.0);
            double z_i = z + offset_w0 + half_L * (k_i / k0_i - 1.0);

            double w_p = beam_w(z_p, k_p, wp);
            double w_s = beam_w(z_s, k_s, ws);
            double w_i = beam_w(z_i, k_i, wi);

            double inv_wp2 = 1.0 / (w_p * w_p);
            double inv_ws2 = 1.0 / (w_s * w_s);
            double inv_wi2 = 1.0 / (w_i * w_i);

            double Re_A = inv_wp2 + inv_ws2 + inv_wi2;
            double Im_A = 0.5 * k_p * inv_R_fn(z_p, k_p, wp)
                        - 0.5 * k_s * inv_R_fn(z_s, k_s, ws)
                        - 0.5 * k_i * inv_R_fn(z_i, k_i, wi);

            double norm_val = 1.0 / (w_p * w_s * w_i);

            double denom   = 2.0 * (Re_A * Re_A + Im_A * Im_A);
            double inv2A_re =  Re_A / denom;
            double inv2A_im =  Im_A / denom;

            double phase = z * dk
                + (-gouy_fn(z_p, k_p, wp)
                   + gouy_fn(z_s, k_s, ws)
                   + gouy_fn(z_i, k_i, wi));

            double sn, cs;
            sincos(phase, &sn, &cs);

            double I_re = norm_val * (cs * inv2A_re - sn * inv2A_im);
            double I_im = norm_val * (sn * inv2A_re + cs * inv2A_im);

            c_re[il] += wz * I_re;
            c_im[il] += wz * I_im;
        }
    }

    double S2 = 0.0;
    {
        double twopi = 2.0 * M_PI;
        for (int il = 0; il < N_SPEC; il++) {
            double re = twopi * c_re[il];
            double im = twopi * c_im[il];
            S2 += w_spec[il] * alpha_sq[il] * (re * re + im * im);
        }
    }
    free(c_re);   /* frees c_im too (single allocation) */


    /* ── Singles: per-spectral-point double-z ─────────────────
     *
     *  The double-z kernel K(z₁,z₂) depends on all three beam
     *  params that vary per spectral point (including pump for
     *  broadband), so everything is computed per spectral point.
     * ────────────────────────────────────────────────────────── */

    double S1_s = 0.0, S1_i = 0.0;

    for (int il = 0; il < N_SPEC; il++) {
        double k_s = k_s_arr[il];
        double k_i = k_i_arr[il];
        double k_p = k_p_arr[il];
        double dk  = dk_arr[il];
        double wt  = w_spec[il] * alpha_sq[il];

        /* Precompute all three beams at all z-nodes for this spectral point */
        double b_wp[MAX_NZ], b_ws[MAX_NZ], b_wi[MAX_NZ];
        double b_wp2[MAX_NZ];
        double b_curv[MAX_NZ], b_base[MAX_NZ];
        double b_gi[MAX_NZ], b_gs[MAX_NZ];

        double off_p_il = offset_w0 + half_L * (k_p / k0_p - 1.0);
        double off_s_il = offset_w0 + half_L * (k_s / k0_s - 1.0);
        double off_i_il = offset_w0 + half_L * (k_i / k0_i - 1.0);

        for (int iz = 0; iz < nz; iz++) {
            double z   = z_nodes[iz];
            double z_p = z + off_p_il;
            double z_s = z + off_s_il;
            double z_i = z + off_i_il;

            b_wp[iz]  = beam_w(z_p, k_p, wp);
            b_ws[iz]  = beam_w(z_s, k_s, ws);
            b_wi[iz]  = beam_w(z_i, k_i, wi);
            b_wp2[iz] = b_wp[iz] * b_wp[iz];

            b_gs[iz] = gouy_fn(z_s, k_s, ws);
            b_gi[iz] = gouy_fn(z_i, k_i, wi);

            double gp  = gouy_fn(z_p, k_p, wp);

            b_curv[iz] = 0.5 * k_p * inv_R_fn(z_p, k_p, wp)
                       - 0.5 * k_s * inv_R_fn(z_s, k_s, ws)
                       - 0.5 * k_i * inv_R_fn(z_i, k_i, wi);

            b_base[iz] = z * dk + (-gp + b_gs[iz] + b_gi[iz]);
        }

        /* Off-diagonal double-z */
        double s1s_acc = 0.0, s1i_acc = 0.0;

        for (int iz1 = 0; iz1 < nz; iz1++) {
            double wz1 = z_weights[iz1] * chi_z[iz1];

            double wp1_2 = b_wp2[iz1];
            double ws1_2 = b_ws[iz1] * b_ws[iz1];
            double wi1_2 = b_wi[iz1] * b_wi[iz1];

            double sig1 = 1.0/wp1_2 + 1.0/ws1_2 + 1.0/wi1_2;
            double N1   = 1.0 / (b_wp[iz1] * b_ws[iz1] * b_wi[iz1]);

            double a1i_re =  sig1 * wi1_2 * 0.5;
            double a1i_im = -b_curv[iz1] * wi1_2 * 0.5;

            double b1s_re =  sig1 * ws1_2 * 0.5;
            double b1s_im = -b_curv[iz1] * ws1_2 * 0.5;

            for (int iz2 = 0; iz2 < nz; iz2++) {
                if (iz2 == iz1) continue;

                double wz2 = z_weights[iz2] * chi_z[iz2];

                double wp2_2 = b_wp2[iz2];
                double ws2_2 = b_ws[iz2] * b_ws[iz2];
                double wi2_2 = b_wi[iz2] * b_wi[iz2];

                double sig2 = 1.0/wp2_2 + 1.0/ws2_2 + 1.0/wi2_2;
                double N2   = 1.0 / (b_wp[iz2] * b_ws[iz2] * b_wi[iz2]);

                double dphi = b_base[iz1] - b_base[iz2];
                double ep_re, ep_im;
                sincos(dphi, &ep_im, &ep_re);

                /* S1_s: trace over idler */
                {
                    double dg = b_gi[iz1] - b_gi[iz2];
                    double s_re, s_im;
                    sincos(2.0 * dg, &s_im, &s_re);

                    double a2c_re =  sig2 * wi2_2 * 0.5;
                    double a2c_im =  b_curv[iz2] * wi2_2 * 0.5;

                    double aa_re, aa_im;
                    cmul(a1i_re, a1i_im, a2c_re, a2c_im, &aa_re, &aa_im);

                    double a1m1_re = a1i_re - 1.0, a1m1_im = a1i_im;
                    double a2m1_re = a2c_re - 1.0, a2m1_im = a2c_im;
                    double mm_re, mm_im;
                    cmul(a1m1_re, a1m1_im, a2m1_re, a2m1_im, &mm_re, &mm_im);

                    double smm_re, smm_im;
                    cmul(s_re, s_im, mm_re, mm_im, &smm_re, &smm_im);

                    double D_re = aa_re - smm_re;
                    double D_im = aa_im - smm_im;

                    double num = N1 * N2 * wi1_2 * wi2_2 / 16.0;
                    double frac_re, frac_im;
                    cdiv(num, 0.0, D_re, D_im, &frac_re, &frac_im);

                    double I_re, I_im;
                    cmul(frac_re, frac_im, ep_re, ep_im, &I_re, &I_im);

                    s1s_acc += wz1 * wz2 * I_re;
                }

                /* S1_i: trace over signal */
                {
                    double dg = b_gs[iz1] - b_gs[iz2];
                    double s_re, s_im;
                    sincos(2.0 * dg, &s_im, &s_re);

                    double b2c_re =  sig2 * ws2_2 * 0.5;
                    double b2c_im =  b_curv[iz2] * ws2_2 * 0.5;

                    double bb_re, bb_im;
                    cmul(b1s_re, b1s_im, b2c_re, b2c_im, &bb_re, &bb_im);

                    double b1m1_re = b1s_re - 1.0, b1m1_im = b1s_im;
                    double b2m1_re = b2c_re - 1.0, b2m1_im = b2c_im;
                    double mm_re, mm_im;
                    cmul(b1m1_re, b1m1_im, b2m1_re, b2m1_im, &mm_re, &mm_im);

                    double smm_re, smm_im;
                    cmul(s_re, s_im, mm_re, mm_im, &smm_re, &smm_im);

                    double D_re = bb_re - smm_re;
                    double D_im = bb_im - smm_im;

                    double num = N1 * N2 * ws1_2 * ws2_2 / 16.0;
                    double frac_re, frac_im;
                    cdiv(num, 0.0, D_re, D_im, &frac_re, &frac_im);

                    double I_re, I_im;
                    cmul(frac_re, frac_im, ep_re, ep_im, &I_re, &I_im);

                    s1i_acc += wz1 * wz2 * I_re;
                }
            }
        }

        /* Diagonal (Parseval) */
        double s1s_diag = 0.0, s1i_diag = 0.0;

        for (int iz = 0; iz < nz; iz++) {
            double ch2 = chi_z[iz] * chi_z[iz];
            double wz2 = z_weights[iz] * z_weights[iz] * ch2;
            double wp2 = b_wp2[iz];
            double ws2 = b_ws[iz] * b_ws[iz];
            double wi2 = b_wi[iz] * b_wi[iz];

            s1s_diag += wz2 / (16.0 * (1.0/wp2 + 1.0/ws2) * wp2 * ws2);
            s1i_diag += wz2 / (16.0 * (1.0/wp2 + 1.0/wi2) * wp2 * wi2);
        }

        double fourpi2 = 4.0 * M_PI * M_PI;
        S1_s += wt * fourpi2 * (s1s_acc + s1s_diag);
        S1_i += wt * fourpi2 * (s1i_acc + s1i_diag);
    }

    *S2_out   = S2;
    *S1_s_out = S1_s;
    *S1_i_out = S1_i;
}


/* ══════════════════════════════════════════════════════════════════
 *  PUBLIC: full waist scan
 *
 *  method = 0 → LG mode sum          (original)
 *  method = 1 → double-z analytic    (SPDCalc-style, batched)
 * ══════════════════════════════════════════════════════════════════ */

EXPORT void scan_waists(
    int N_WP, int N_WS,
    const double *wp_arr, const double *ws_arr,
    int N_SPEC,
    const double *k_s_arr, const double *k_i_arr,
    const double *dk_arr,  const double *w_spec,
    const double *alpha_sq,
    const double *k_p_arr,
    double k0_p, double L, double offset_w0,
    double lam0_p, double lam0_s, double lam0_i,
    double n0_p, double n0_s, double n0_i,
    double k0_s, double k0_i,
    int P_MAX_MIN,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    int N_R, const double *r_ref,   const double *r_weights,
    int method,
    double *S2_map, double *H_map)
{
    int total = N_WS * N_WP;
    int nz = N_Z < MAX_NZ ? N_Z : MAX_NZ;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int flat = 0; flat < total; flat++) {
        int iw_s = flat / N_WP;
        int iw_p = flat % N_WP;

        double wp_val = wp_arr[iw_p];
        double ws_val = ws_arr[iw_s];
        double wi_val = ws_val;

        double S2 = 0.0, S1_s = 0.0, S1_i = 0.0;

        if (method == 0) {
            /* ── LG mode sum ──────────────────────────────── */
            int PM = adaptive_pmax(wp_val, ws_val, wi_val, L,
                                   lam0_p, lam0_s, lam0_i,
                                   n0_p, n0_s, n0_i, P_MAX_MIN);
            if (PM > MAX_P) PM = MAX_P;

            double P_0i[MAX_P+1], P_i0[MAX_P+1];

            for (int il = 0; il < N_SPEC; il++) {
                one_spectral_point_lgsum(
                    k_s_arr[il], k_i_arr[il], dk_arr[il], PM,
                    wp_val, ws_val, wi_val,
                    k_p_arr[il], k0_p, L, offset_w0,
                    k0_s, k0_i,
                    N_Z, z_nodes, z_weights, chi_z,
                    N_R, r_ref, r_weights,
                    P_0i, P_i0);

                double w  = w_spec[il] * alpha_sq[il];
                double ss = 0.0, si = 0.0;
                for (int i = 0; i <= PM; i++) { ss += P_0i[i]; si += P_i0[i]; }
                S2   += w * P_0i[0];
                S1_s += w * ss;
                S1_i += w * si;
            }
        } else {
            /* ── Double-z analytic trace (batched) ───────── */
            batch_doublez_one_waist(
                N_SPEC, k_s_arr, k_i_arr, dk_arr, w_spec, alpha_sq,
                k_p_arr,
                wp_val, ws_val, wi_val,
                k0_p, L, offset_w0,
                k0_s, k0_i,
                nz, z_nodes, z_weights, chi_z,
                &S2, &S1_s, &S1_i);
        }

        double H_s = (S1_i > 0.0) ? S2 / S1_i : 0.0;
        double H_i = (S1_s > 0.0) ? S2 / S1_s : 0.0;

        int idx = iw_s * N_WP + iw_p;
        S2_map[idx] = S2;
        H_map[idx]  = sqrt(H_s * H_i);
    }
}


/* ── Single-point API ─────────────────────────────────────────── */

EXPORT void compute_single(
    int N_SPEC,
    const double *k_s_arr, const double *k_i_arr,
    const double *dk_arr,  const double *w_spec,
    const double *alpha_sq,
    const double *k_p_arr,
    int P_MAX,
    double wp, double ws, double wi,
    double k0_p, double L, double offset_w0,
    double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    int N_R, const double *r_ref,   const double *r_weights,
    int method,
    double *S2_out, double *S1_s_out, double *S1_i_out)
{
    int pm = P_MAX < MAX_P ? P_MAX : MAX_P;
    double S2 = 0.0, S1_s = 0.0, S1_i = 0.0;

    if (method == 0) {
        double P_0i[MAX_P+1], P_i0[MAX_P+1];

        for (int il = 0; il < N_SPEC; il++) {
            one_spectral_point_lgsum(
                k_s_arr[il], k_i_arr[il], dk_arr[il], pm,
                wp, ws, wi,
                k_p_arr[il], k0_p, L, offset_w0,
                k0_s, k0_i,
                N_Z, z_nodes, z_weights, chi_z,
                N_R, r_ref, r_weights,
                P_0i, P_i0);

            double w = w_spec[il] * alpha_sq[il];
            double ss = 0.0, si = 0.0;
            for (int i = 0; i <= pm; i++) { ss += P_0i[i]; si += P_i0[i]; }
            S2   += w * P_0i[0];
            S1_s += w * ss;
            S1_i += w * si;
        }
    } else {
        int nz = N_Z < MAX_NZ ? N_Z : MAX_NZ;
        batch_doublez_one_waist(
            N_SPEC, k_s_arr, k_i_arr, dk_arr, w_spec, alpha_sq,
            k_p_arr,
            wp, ws, wi,
            k0_p, L, offset_w0,
            k0_s, k0_i,
            nz, z_nodes, z_weights, chi_z,
            &S2, &S1_s, &S1_i);
    }

    *S2_out   = S2;
    *S1_s_out = S1_s;
    *S1_i_out = S1_i;
}

/* ══════════════════════════════════════════════════════════════════
 *  PUBLIC: C₀₀ heatmap — compute |C₀₀|² for a flat array of
 *  spectral points.  Designed for spectrum-vs-T heatmaps.
 *
 *  Each point j has its own (k_s, k_i, k_p, dk).
 *  The z-integral uses the analytical transverse overlap (no r-grid).
 *  OpenMP parallelisation over the flat index.
 * ══════════════════════════════════════════════════════════════════ */

EXPORT void compute_c00_heatmap(
    int N_total,
    const double *k_s_arr,
    const double *k_i_arr,
    const double *k_p_arr,
    const double *dk_arr,
    double wp, double ws, double wi,
    double L, double offset_w0,
    double k0_p, double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    double *c00sq_out)
{
    int nz = N_Z < MAX_NZ ? N_Z : MAX_NZ;
    double half_L = 0.5 * L;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int j = 0; j < N_total; j++) {
        double k_s  = k_s_arr[j];
        double k_i  = k_i_arr[j];
        double k_p  = k_p_arr[j];
        double dk   = dk_arr[j];

        /* Per-beam wavelength-dependent waist offsets (Eq. 47) */
        double off_p = offset_w0 + half_L * (k_p / k0_p - 1.0);
        double off_s = offset_w0 + half_L * (k_s / k0_s - 1.0);
        double off_i = offset_w0 + half_L * (k_i / k0_i - 1.0);

        double acc_re = 0.0, acc_im = 0.0;

        for (int iz = 0; iz < nz; iz++) {
            double z  = z_nodes[iz];
            double wz = z_weights[iz] * chi_z[iz];

            double z_p = z + off_p;
            double z_s = z + off_s;
            double z_i = z + off_i;

            double w_p = beam_w(z_p, k_p, wp);
            double w_s = beam_w(z_s, k_s, ws);
            double w_i = beam_w(z_i, k_i, wi);

            double Re_A = 1.0/(w_p*w_p) + 1.0/(w_s*w_s) + 1.0/(w_i*w_i);
            double Im_A = 0.5*k_p*inv_R_fn(z_p, k_p, wp)
                        - 0.5*k_s*inv_R_fn(z_s, k_s, ws)
                        - 0.5*k_i*inv_R_fn(z_i, k_i, wi);

            double norm_val = 1.0 / (w_p * w_s * w_i);
            double denom = 2.0 * (Re_A*Re_A + Im_A*Im_A);
            double inv2A_re = Re_A / denom;
            double inv2A_im = Im_A / denom;

            double phase = z * dk
                + (-gouy_fn(z_p, k_p, wp)
                   + gouy_fn(z_s, k_s, ws)
                   + gouy_fn(z_i, k_i, wi));

            double sn, cs;
            sincos(phase, &sn, &cs);

            double I_re = norm_val * (cs * inv2A_re - sn * inv2A_im);
            double I_im = norm_val * (sn * inv2A_re + cs * inv2A_im);

            acc_re += wz * I_re;
            acc_im += wz * I_im;
        }

        double twopi = 2.0 * M_PI;
        double re = twopi * acc_re;
        double im = twopi * acc_im;
        c00sq_out[j] = re*re + im*im;
    }
}


/* ══════════════════════════════════════════════════════════════════
 *  PUBLIC: Mode-resolved spectra with λ_i integration
 *
 *  For each λ_s block (N_i idler quadrature points), computes the
 *  LG-mode-resolved amplitudes |C_{0,pi}|² and |C_{ps,0}|² at every
 *  idler point, then accumulates the weighted sum:
 *
 *    P_0i_out[p * N_lam + il] = Σ_j  weights[il*N_i + j] · |C_{0,p}|²_j
 *    P_i0_out[p * N_lam + il] = Σ_j  weights[il*N_i + j] · |C_{p,0}|²_j
 *
 *  weights[j] should include dλ_i × trapezoidal shape × α²(λ_p).
 *
 *  OpenMP parallelisation is over the N_lam λ_s blocks.
 * ══════════════════════════════════════════════════════════════════ */

EXPORT void compute_modes_spectrum(
    int N_lam, int N_i, int P_MAX,
    const double *k_s_arr,       /* (N_lam * N_i,) */
    const double *k_i_arr,       /* (N_lam * N_i,) */
    const double *k_p_arr,       /* (N_lam * N_i,) */
    const double *dk_arr,        /* (N_lam * N_i,) */
    const double *weights,       /* (N_lam * N_i,) = dλ_i × trap × α² */
    double wp, double ws, double wi,
    double k0_p, double L, double offset_w0,
    double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    int N_R, const double *r_ref, const double *r_weights,
    double *P_0i_out,            /* ((P_MAX+1) * N_lam,) row-major */
    double *P_i0_out)            /* ((P_MAX+1) * N_lam,) row-major */
{
    int pm = P_MAX < MAX_P ? P_MAX : MAX_P;
    int n_modes = pm + 1;

    /* Zero the output arrays */
    memset(P_0i_out, 0, n_modes * N_lam * sizeof(double));
    memset(P_i0_out, 0, n_modes * N_lam * sizeof(double));

    #pragma omp parallel for schedule(dynamic, 1)
    for (int il = 0; il < N_lam; il++) {
        int base = il * N_i;

        double local_P0i[MAX_P+1];
        double local_Pi0[MAX_P+1];
        double acc_P0i[MAX_P+1];
        double acc_Pi0[MAX_P+1];
        memset(acc_P0i, 0, n_modes * sizeof(double));
        memset(acc_Pi0, 0, n_modes * sizeof(double));

        for (int ji = 0; ji < N_i; ji++) {
            int idx = base + ji;
            double w = weights[idx];
            if (w <= 0.0) continue;

            one_spectral_point_lgsum(
                k_s_arr[idx], k_i_arr[idx], dk_arr[idx], pm,
                wp, ws, wi,
                k_p_arr[idx], k0_p, L, offset_w0,
                k0_s, k0_i,
                N_Z, z_nodes, z_weights, chi_z,
                N_R, r_ref, r_weights,
                local_P0i, local_Pi0);

            for (int p = 0; p < n_modes; p++) {
                acc_P0i[p] += w * local_P0i[p];
                acc_Pi0[p] += w * local_Pi0[p];
            }
        }

        for (int p = 0; p < n_modes; p++) {
            P_0i_out[p * N_lam + il] = acc_P0i[p];
            P_i0_out[p * N_lam + il] = acc_Pi0[p];
        }
    }
}


/* ══════════════════════════════════════════════════════════════════
 *  PUBLIC: Integrated C₀₀ spectrum — λ_i summation inside C
 *
 *  The input arrays are structured as N_outer blocks of N_i points.
 *  For each outer index (T, λ_s), computes:
 *
 *    spectrum_out[j] = Σ_i  weights[j*N_i + i] · |C₀₀(j*N_i + i)|²
 *
 *  This avoids allocating the large intermediate |C₀₀|² array and
 *  the Python reduction step.  OpenMP over the N_outer blocks.
 * ══════════════════════════════════════════════════════════════════ */

EXPORT void compute_c00_integrated(
    int N_outer, int N_i,
    const double *k_s_arr,       /* (N_outer * N_i,) */
    const double *k_i_arr,
    const double *k_p_arr,
    const double *dk_arr,
    const double *weights,       /* (N_outer * N_i,) = dλ_i × trap × α² */
    double wp, double ws, double wi,
    double L, double offset_w0,
    double k0_p, double k0_s, double k0_i,
    int N_Z, const double *z_nodes, const double *z_weights,
    const double *chi_z,
    double *spectrum_out)        /* (N_outer,) */
{
    int nz = N_Z < MAX_NZ ? N_Z : MAX_NZ;
    double half_L = 0.5 * L;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int j = 0; j < N_outer; j++) {
        int base = j * N_i;
        double acc_spec = 0.0;

        for (int ii = 0; ii < N_i; ii++) {
            int idx = base + ii;
            double w = weights[idx];
            if (w <= 0.0) continue;

            double k_s = k_s_arr[idx];
            double k_i = k_i_arr[idx];
            double k_p = k_p_arr[idx];
            double dk  = dk_arr[idx];

            double off_p = offset_w0 + half_L * (k_p / k0_p - 1.0);
            double off_s = offset_w0 + half_L * (k_s / k0_s - 1.0);
            double off_i = offset_w0 + half_L * (k_i / k0_i - 1.0);

            double acc_re = 0.0, acc_im = 0.0;

            for (int iz = 0; iz < nz; iz++) {
                double z  = z_nodes[iz];
                double wz = z_weights[iz] * chi_z[iz];

                double z_p = z + off_p;
                double z_s = z + off_s;
                double z_i = z + off_i;

                double w_p = beam_w(z_p, k_p, wp);
                double w_s = beam_w(z_s, k_s, ws);
                double w_i = beam_w(z_i, k_i, wi);

                double Re_A = 1.0/(w_p*w_p) + 1.0/(w_s*w_s) + 1.0/(w_i*w_i);
                double Im_A = 0.5*k_p*inv_R_fn(z_p, k_p, wp)
                            - 0.5*k_s*inv_R_fn(z_s, k_s, ws)
                            - 0.5*k_i*inv_R_fn(z_i, k_i, wi);

                double norm_val = 1.0 / (w_p * w_s * w_i);
                double denom = 2.0 * (Re_A*Re_A + Im_A*Im_A);
                double inv2A_re = Re_A / denom;
                double inv2A_im = Im_A / denom;

                double phase = z * dk
                    + (-gouy_fn(z_p, k_p, wp)
                       + gouy_fn(z_s, k_s, ws)
                       + gouy_fn(z_i, k_i, wi));

                double sn, cs;
                sincos(phase, &sn, &cs);

                double I_re = norm_val * (cs * inv2A_re - sn * inv2A_im);
                double I_im = norm_val * (sn * inv2A_re + cs * inv2A_im);

                acc_re += wz * I_re;
                acc_im += wz * I_im;
            }

            double twopi = 2.0 * M_PI;
            double re = twopi * acc_re;
            double im = twopi * acc_im;
            acc_spec += w * (re*re + im*im);
        }

        spectrum_out[j] = acc_spec;
    }
}
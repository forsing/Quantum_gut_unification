# Quantum_gut_unification 



import random
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
SHORT_WIN = 2
MID_WIN = 6
LONG_WIN = 18
UNIFICATION_SCALES = (0.20, 0.40, 0.60, 0.80, 0.95)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < LONG_WIN + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def extract_step_features(
    w_short: np.ndarray,
    w_mid: np.ndarray,
    w_long: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1_feat = np.concatenate([w_short[-1], w_short[-1] - w_short[-2]])

    mid_mean = np.mean(w_mid, axis=0)
    mid_doublets = np.diff(w_mid[-1], n=1)
    su2_feat = np.concatenate([mid_mean, mid_doublets])

    long_mean = np.mean(w_long, axis=0)
    long_std = np.std(w_long, axis=0)
    triplets = np.diff(w_long[-1], n=2)
    su3_feat = np.concatenate([long_mean, long_std, triplets])

    return u1_feat, su2_feat, su3_feat


def build_three_force_features(
    draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    u1_list = []
    su2_list = []
    su3_list = []
    y_list = []

    norm_draws = draws.astype(np.float64) / MAX_NUM
    for t in range(LONG_WIN, n_total):
        u1_f, su2_f, su3_f = extract_step_features(
            norm_draws[t - SHORT_WIN : t],
            norm_draws[t - MID_WIN : t],
            norm_draws[t - LONG_WIN : t],
        )
        u1_list.append(u1_f)
        su2_list.append(su2_f)
        su3_list.append(su3_f)
        y_list.append(draws[t])

    return (
        np.array(u1_list, dtype=np.float64),
        np.array(su2_list, dtype=np.float64),
        np.array(su3_list, dtype=np.float64),
        np.array(y_list, dtype=np.int64),
    )


def fit_single_force_propagator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma: float,
    reg_lambda: float,
) -> np.ndarray:
    phi = np.hstack([np.ones((len(x_train), 1), dtype=np.float64), x_train])
    n_features = int(phi.shape.__getitem__(1))
    n_samples = len(y_train)

    target_dist = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    grid = np.arange(1, MAX_NUM + 1, dtype=np.float64)[None, :]
    for pos in range(NUM_POSITIONS):
        y_col = y_train[:, pos][:, None].astype(np.float64)
        kernel = np.exp(-0.5 * ((grid - y_col) / sigma) ** 2)
        kernel /= np.sum(kernel, axis=1, keepdims=True)
        target_dist[:, pos, :] = kernel

    y_flat = target_dist.reshape(n_samples, NUM_POSITIONS * MAX_NUM)
    gram = phi.T @ phi + reg_lambda * np.eye(n_features, dtype=np.float64)
    return np.linalg.solve(gram, phi.T @ y_flat)


def predict_force_dist(weights: np.ndarray, x_step: np.ndarray) -> np.ndarray:
    phi_step = np.concatenate([np.array([1.0], dtype=np.float64), x_step])
    pred = (phi_step @ weights).reshape(NUM_POSITIONS, MAX_NUM)
    pred = np.maximum(pred, 1e-8)
    pred /= np.sum(pred, axis=1, keepdims=True)
    return pred


def running_couplings(lambda_scale: float) -> tuple[float, float, float]:
    alpha_1 = 0.25 + 0.25 * lambda_scale
    alpha_2 = 0.45 - 0.10 * lambda_scale
    alpha_3 = 0.85 - 0.50 * lambda_scale
    total = alpha_1 + alpha_2 + alpha_3
    return alpha_1 / total, alpha_2 / total, alpha_3 / total


def build_position_gut_circuit(
    c1: float,
    c2: float,
    c3: float,
    e1: float,
    e2: float,
    e3: float,
) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)

    for q in range(NUM_QUBITS):
        qc.rz(c1 * e1 * np.pi * (q + 1) / NUM_QUBITS, q)

    for q in range(NUM_QUBITS):
        qc.ry(c2 * e2 * 0.6 * np.pi, q)
    for q in range(0, NUM_QUBITS - 1, 2):
        qc.crx(c2 * e2 * 0.5 * np.pi, q, q + 1)

    for q in range(NUM_QUBITS - 1):
        qc.rzz(c3 * e3 * 0.4 * np.pi, q, q + 1)
    for q in range(NUM_QUBITS - 2):
        qc.rzz(c3 * e3 * 0.25 * np.pi, q, q + 2)

    return qc


def evolve_gut_quantum_register(
    p_u1: np.ndarray,
    p_su2: np.ndarray,
    p_su3: np.ndarray,
    c1: float,
    c2: float,
    c3: float,
) -> np.ndarray:
    unified_prob = c1 * p_u1 + c2 * p_su2 + c3 * p_su3
    unified_prob = np.maximum(unified_prob, 1e-12)
    unified_prob /= np.sum(unified_prob)

    e1 = float(np.sum(p_u1 ** 2) * MAX_NUM)
    e2 = float(np.sum(p_su2 ** 2) * MAX_NUM)
    e3 = float(np.sum(p_su3 ** 2) * MAX_NUM)

    log_diff_12 = np.log(np.maximum(p_u1, 1e-8) / np.maximum(p_su2, 1e-8))
    log_diff_32 = np.log(np.maximum(p_su3, 1e-8) / np.maximum(p_su2, 1e-8))
    phase_profile = np.pi * np.tanh(0.5 * (c1 * log_diff_12 + c3 * log_diff_32))

    init_state = np.zeros(STATE_DIM, dtype=np.complex128)
    init_state[1 : MAX_NUM + 1] = np.sqrt(unified_prob) * np.exp(1j * phase_profile)
    norm = float(np.linalg.norm(init_state))
    if norm <= 1e-15:
        raise ValueError("Nulta norma GUT kvantnog stanja.")
    init_state /= norm
    if abs(float(np.linalg.norm(init_state)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji inicijalnih kvantnih amplituda.")

    gut_qc = build_position_gut_circuit(c1, c2, c3, e1, e2, e3)
    sv = Statevector(init_state).evolve(gut_qc)
    full_probs = sv.probabilities()
    valid_probs = full_probs[1 : MAX_NUM + 1]
    valid_probs = np.maximum(valid_probs, 1e-12)
    valid_probs /= np.sum(valid_probs)

    if abs(float(np.sum(valid_probs)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji izlazne GUT raspodele.")
    return valid_probs


def decode_combination(q_probs: np.ndarray) -> list[int]:
    available = np.ones(MAX_NUM, dtype=bool)
    selected = []
    for pos in range(NUM_POSITIONS):
        masked = np.where(available, q_probs[pos], -1.0)
        best_idx = int(np.argmax(masked))
        selected.append(best_idx + 1)
        available[best_idx] = False
    return sorted(selected)


def evaluate_gut_range(
    w_u1: np.ndarray,
    w_su2: np.ndarray,
    w_su3: np.ndarray,
    x_u1: np.ndarray,
    x_su2: np.ndarray,
    x_su3: np.ndarray,
    y_arr: np.ndarray,
    start_idx: int,
    end_idx: int,
    lambda_scale: float,
) -> float:
    c1, c2, c3 = running_couplings(lambda_scale)

    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        p_u1 = predict_force_dist(w_u1, x_u1[t])
        p_su2 = predict_force_dist(w_su2, x_su2[t])
        p_su3 = predict_force_dist(w_su3, x_su3[t])
        for pos in range(NUM_POSITIONS):
            q_dist = evolve_gut_quantum_register(
                p_u1[pos],
                p_su2[pos],
                p_su3[pos],
                c1,
                c2,
                c3,
            )
            target_num = int(y_arr[t, pos])
            p = max(float(q_dist[target_num - 1]), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    x_u1, x_su2, x_su3, y_all = build_three_force_features(draws)
    n_samples = len(y_all)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    w_u1_tr = fit_single_force_propagator(x_u1[:train_end], y_all[:train_end], sigma=1.0, reg_lambda=0.05)
    w_su2_tr = fit_single_force_propagator(x_su2[:train_end], y_all[:train_end], sigma=1.4, reg_lambda=0.15)
    w_su3_tr = fit_single_force_propagator(x_su3[:train_end], y_all[:train_end], sigma=1.9, reg_lambda=0.35)

    best_lambda = float(next(iter(UNIFICATION_SCALES)))
    best_val_nll = float("inf")

    for lam in UNIFICATION_SCALES:
        val_nll = evaluate_gut_range(
            w_u1_tr,
            w_su2_tr,
            w_su3_tr,
            x_u1,
            x_su2,
            x_su3,
            y_all,
            train_end,
            val_end,
            float(lam),
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_lambda = float(lam)

    w_u1_tv = fit_single_force_propagator(x_u1[:val_end], y_all[:val_end], sigma=1.0, reg_lambda=0.05)
    w_su2_tv = fit_single_force_propagator(x_su2[:val_end], y_all[:val_end], sigma=1.4, reg_lambda=0.15)
    w_su3_tv = fit_single_force_propagator(x_su3[:val_end], y_all[:val_end], sigma=1.9, reg_lambda=0.35)

    test_nll = evaluate_gut_range(
        w_u1_tv,
        w_su2_tv,
        w_su3_tv,
        x_u1,
        x_su2,
        x_su3,
        y_all,
        val_end,
        n_samples,
        best_lambda,
    )

    w_u1_full = fit_single_force_propagator(x_u1, y_all, sigma=1.0, reg_lambda=0.05)
    w_su2_full = fit_single_force_propagator(x_su2, y_all, sigma=1.4, reg_lambda=0.15)
    w_su3_full = fit_single_force_propagator(x_su3, y_all, sigma=1.9, reg_lambda=0.35)

    norm_draws = draws.astype(np.float64) / MAX_NUM
    u1_last, su2_last, su3_last = extract_step_features(
        norm_draws[-SHORT_WIN:],
        norm_draws[-MID_WIN:],
        norm_draws[-LONG_WIN:],
    )


    p_u1_next = predict_force_dist(w_u1_full, u1_last)
    p_su2_next = predict_force_dist(w_su2_full, su2_last)
    p_su3_next = predict_force_dist(w_su3_full, su3_last)

    c1_opt, c2_opt, c3_opt = running_couplings(best_lambda)

    final_q_probs = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    for pos in range(NUM_POSITIONS):
        final_q_probs[pos] = evolve_gut_quantum_register(
            p_u1_next[pos],
            p_su2_next[pos],
            p_su3_next[pos],
            c1_opt,
            c2_opt,
            c3_opt,
        )

    next_comb = decode_combination(final_q_probs)
    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (Lambda={best_lambda}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Grand Unification (GUT) Three-Force Running Coupling (7x6 qubita)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Grand Unification (GUT) Three-Force Running Coupling (7x6 qubita)

CSV: loto7_4718_k90.csv
Validacija NLL (Lambda=0.8): 34.7180
Test NLL: 35.2135
NEXT: [5, x, 17, y, 23, z, 33]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (Lambda=0.8): 34.2555
Test NLL: 34.9411
NEXT: [1, x, 7, y, 23, z, 33]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (Lambda=0.8): 35.3685
Test NLL: 35.0026
NEXT: [5, x, 17, y, 25, z, 33]
"""



"""
three quantum forces all unify together 
Quarks 
Leptons 
Boson 
Forces 


Ujedinjenje tri kvantne sile (Velika unifikacija — GUT, kvarkovi, leptoni, bozoni i konstante sprege)
Ujedinjenje elektromagnetne, slabe i jake nuklearne sile na visokim energijama, materija vs. nosioci sile 

	1. Tri skale interakcije koje konvergiraju u jednu tačku (Running Coupling Constants): U fizici velikog ujedinjenja (GUT), tri sile imaju različite jačine na niskim energijama, ali kako skala energije raste, njihove konstante sprege (α₁, α₂, α₃) teku kroz renormalizacionu grupu i spajaju se u jednoj tački unifikacije. U Loto 7/39 nizu to se modeluje kao višeskalna dinamika na tri vremenska horizonta (kratkoročni, srednjoročni i dugoročni prelazni tok) čiji se doprinosi spajaju na energetskoj skali unifikacije Λ radi formiranja zajedničke raspodele verovatnoće.
	2. Podela na fermione (materija) i bozone (nosioci interakcije): Kvarkovi i leptoni predstavljaju stanja sistema (same izvučene brojeve na 7 pozicija), dok baždarni bozoni prenose interakciju između pozicija (međupozicioni prelazi i simetrije SU(3) x SU(2) x U(1)).
	3. Kvantna realizacija putem SU(2) i U(1) rotacija i renormalizacionog toka (Qiskit 1.4.4, 7x6 qubita): Tri sile se kodiraju kao tri sloja unitarnih operatora nad 6-qubitnim registrom za svaku poziciju: U(1) fazni sloj (RZ), SU(2) slabi izospinski sloj (RY + parno uplitanje suseda), i SU(3) jaki sloj (RZZ tročlane/lančane interakcije), čije se jačine sprege spajaju na skali Λ izabranoj hronološkom validacijom.

Princip renormalizacionog toka tri različite vremenske/interakcione skale koje se spajaju u jednu unitarnu kvantnu evoluciju daje potpuno nov, matematički čist model različit od kernel gustine i Maxwellovih talasa.

Princip velikog ujedinjenja tri kvantne sile (GUT — renormalizacioni tok tri skale interakcije) za Loto 7/39.

• Tri skale sprege (U(1), SU(2), SU(3)): Iz hronološkog niza kola računaju se tri različite vremenske i interakcione skale:
	1. U(1) (elektromagnetni / kratkoročni tok): Neposredni prelaz iz poslednjeg kola (t - 1).
	2. SU(2) (slabi / srednjoročni izospinski tok): Eksponencijalno ponderisana dinamika parova pozicija kroz srednji prozor (k = 6 kola).
	3. SU(3) (jaki / dugoročni kolorni tok): Tročlane korelacije i dugoročni strukturni režim (k = 18 kola).
• Renormalizaciono ujedinjenje na skali \(\Lambda \): Hiperparametar energetske skale \(\Lambda \) određuje kako se konstante sprege \(\alpha_1(\Lambda), \alpha_2(\Lambda), \alpha_3(\Lambda)\) spajaju u zajedničku raspodelu amplituda.
• Kvantno kolo za 3 baždarne simetrije (Qiskit 1.4.4, 7x6 qubita): Svaki registar (6 qubita, stanja \(1\dots39\)) prolazi kroz tri uzastopna unitarna bloka: U(1) fazni blok (RZ), SU(2) slabi blok (RY + CRX parno uplitanje) i SU(3) jaki blok (RZZ tročlane veze), nakon čega se očitavaju kvantne verovatnoće \(\vert{}\alpha_i\vert{}^2\).

Velika unifikacija tri kvantne sile — GUT. 
Three Quantum Forces / GUT.
"""

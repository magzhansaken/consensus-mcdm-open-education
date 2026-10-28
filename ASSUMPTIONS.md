# Table of assumptions (source of Table S1 of the associated article)

Every number that is not a published measurement comes from one of the rows below. Rows marked **assumption** are
declared choices; rows marked **documented** cite a specification or configuration; rows marked **calibrated**
are fitted on the anchors in `data/anchors.csv`.

| ID | Item | Value | Kind | Basis |
|---|---|---|---|---|
| A1 | Consortium size | n = 7 organisations, one consensus node + one executing node each | assumption | odd n: f_byz = 2 (BFT), f_crash = 3 (Raft); sensitivity n = 4, 10 |
| A2 | Network | campus LAN, RTT 2 ms (log-normal, CV 0.2), 1 Gbit/s links; block transfer time = bytes × fan-out / bandwidth (no per-connection window cap); sensitivity RTT 20 ms | assumption | typical campus backbone |
| A3 | Node | 4 vCPU; utilisation = CPU-seconds / (4 × wall-clock) | assumption | |
| A4 | Transaction | 1 000 B payload + platform envelope: Fabric 2 500 B, Besu 200 B, CometBFT 100 B, HotStuff 100 B | assumption | Fabric envelopes carry endorsements and certificates |
| A5 | Application tier | commit service c_block + c_tx·B and 6.0 ms CPU per tx per executing node, calibrated on Fabric (per-role CPU of Husna et al. 2026) and applied to CometBFT and HotStuff, whose anchors have no execution layer; Besu anchors include EVM execution | calibrated + assumption | role tables of Husna et al. 2026 (peer + chaincode + CouchDB ≈ 6 ms CPU/tx at 50–150 TPS) |
| A6 | Block formation | Fabric: MaxMessageCount 100, BatchTimeout 2 s; Besu: block period 2 s, txpool 4 096 tx; CometBFT: timeout_commit 1 s, mempool 5 000 tx (block ≤ 20 000 tx); HotStuff: proposer-driven, a new proposal as soon as the 2-chain pipeline slot frees (≥ 100 ms batch delay), all pending batches | documented | channel configuration of Husna et al. 2026; Besu/CometBFT docs; HotStuff benchmark settings |
| A7 | Besu capacity | QBFT, IBFT 2.0: 220 tx per 2-s block (110 TPS burst capacity); Clique: ≥ 423 tx per block (no backlog observed, lower bound) | calibrated | Özara et al. 2026 |
| A8 | Besu execution constant | o = 0.2 ms per tx (shared EVM tier) | calibrated | Clique median latency 1.0 s at a 2-s block period |
| A9 | Leader-failure handling | Raft election 5 s (TickInterval 500 ms × ElectionTick 10); SmartBFT ViewChangeTimeout 20 s; IBFT 2.0/QBFT round change 4 s (requesttimeoutseconds = 2 × blockperiodseconds, as in the Besu tutorial genesis; the documented default is 1 s and the configuration-reference example 10 s); CometBFT timeout_propose 3 s; HotStuff timeout_delay 5 s; Clique out-of-turn sealing 1 s | documented | Fabric sample configtx and ordering-service docs; Fabric 3 BFT configtx (ViewChangeTimeout 20s); Besu QBFT tutorial and configuration reference; CometBFT config/config.go defaults; HotStuff benchmark/fabfile.py |
| A10 | Proposer rotation | Raft stable leader; SmartBFT every 3 decisions; IBFT 2.0, QBFT, CometBFT, HotStuff, Clique every block | documented | library/protocol defaults |
| A11 | Finality | deterministic at commit for all but Clique; Clique after n/2 + 1 = 4 further blocks | documented | Clique is a probabilistic-finality PoA |
| A12 | Ledger overhead | header 400–700 B; signature 64–72 B (ECDSA/Ed25519); Fabric block metadata signature 300 B; BFT quorum certificates carry 2f + 1 signatures | documented | protocol encodings |
| A13 | Fault scenario (C7) | leader crash at 120 s and a second node crash at 300 s of a 600-s run at the deadline-day peak; no operator intervention; with n = 4 only the first crash | assumption | the injected crashes are tolerated by every protocol (n = 7 and 10: two; n = 4: one) |
| E1 | Energy proxy (C4) | 15 W per fully loaded vCPU; idle power excluded (equal for all) | assumption | |
| E2 | Cost (C11) | 0.04 USD per vCPU-hour, 0.10 USD per GB-month; nodes sized at 70 % of peak utilisation | assumption | generic public cloud list prices |
| W1–W7 | Workload (synthetic) | 100 000 active learners × 12 anchored events/day; 5 000 IoT devices, one record per 2 min; two daily peaks (10–13 h, 19–23 h); deadline-day evening ×2.5; enrolment hour +120 tx/s; 1 000 B/tx | assumption (synthetic) | shape after open-university VLE activity patterns (Kuzilek et al. 2017), no OULAD records used |
| L1–L3 | Load levels | nominal 55.6 tx/s (daily mean), peak 108.3 tx/s (deadline-day busiest hour), stress 185 tx/s (enrolment hour) | derived from W1–W7 | |
| R1 | CometBFT transport | per-tx mempool cost o and gossip hop delay g calibrated at 175 validators (fan-out-10 gossip: 3 hops at n = 175, 1 hop at n ≤ 10); vote verification scales with n, mempool cost does not | calibrated + assumption | CometBFT QA v1 report |
| R2 | HotStuff anchor network | AWS multi-region WAN assumed RTT 200 ms (CV 0.2), 10 Gbit/s instances; rounds = batch dissemination + 2-chain commit (3), pipeline depth 2; proposals carry batch digests (fan-out 2) | assumption | HotStuff benchmark documentation (geo-replicated runs) |
| R3 | Calibration run protocol | Fabric fits use 90-s runs with 30-s warm-up (as in Husna et al. 2026); stage 1 fits c_block, c_tx, a, o, m on the batch × load campaign (n = 4); stage 2 fits ν (growth of per-tx work per added validator) on the orderer-count campaign; κ from measured orderer CPU; 3 seeds per point for validation; RTT and crash-fault campaigns are hold-outs | protocol | |
| B1 | Besu burst plausibility check | client submission rate 150 tx/s (not reported in the anchor) | assumption | used only to report implied p50/p95 |
| M1 | Weights | Entropy and CRITIC on the mean matrix; base weights = normalised geometric mean | method | project design |
| M2 | Rankings | TOPSIS (vector normalisation), VIKOR (v = 0.5), COPRAS, EDAS | method | |
| M3 | SMAA-2 | 10 000 weight draws (uniform on the simplex; and Dirichlet centred on the base weights, concentration 50) × 30 seed matrices; aggregation by each ranking method | method | Lahdelma & Salminen 2001 |
| M4 | Seeds and CI | 30 seeds per alternative; 95 % CI by Student t | protocol | |
| O1 | Ordinal scores C8 (attack resistance, 0–4) | one point each for: BFT safety (f < n/3), deterministic finality, proposer rotation, built-in equivocation evidence. Raft 1, Clique 1, IBFT 2.0 3, QBFT 3, SmartBFT 3, HotStuff 3, CometBFT 4 | documented | protocol specifications (Table 2 of the article) |
| O2 | Ordinal scores C10 (manageability, 0–4) | one point each for: validator-set change without restart, contract/config-based validator selection or an LTS release line, metrics endpoint, Caliper/benchmark adapter. Raft 4, SmartBFT 4, QBFT 4, IBFT 2.0 3 (validator change only by on-chain voting), Clique 3 (removed in Besu 26.4), CometBFT 3, HotStuff 0 (research implementation) | documented | Fabric, Besu, CometBFT documentation |
| O3 | IBFT 2.0 vs QBFT | the performance anchor does not separate the two (same block period, same capacity, same round-change timer); they differ in the model only through C10 | limitation | Özara et al. 2026 |
| D1 | Deployment ablations (sensitivity scenarios) | n = 4; n = 10; RTT 20 ms; no shared application tier; operator removes a crashed validator from rotation after its first detected failure (`exclude_dead`); HotStuff with a 1-s proposal interval; Besu round-change timer 10 s instead of 4 s — each with 10 seeds and its own Entropy × CRITIC weights | protocol | |

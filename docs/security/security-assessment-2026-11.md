<!-- markdownlint-disable MD013 -->
# Dragonfly fork: security assessment and feature-gap analysis

Date: 2026-11-07. Tree: `f8ca066` on `main`. Scope: the whole repository, with extra depth on the
fork-specific `datacontroller/` operator (commit `d59f58a`), which is the only code in this tree
that does not come from upstream `dragonflyoss/dragonfly`.

Method: five parallel source reviews (datacontroller, manager, scheduler and shared packages,
supply chain and deployment, feature gaps), each claim re-verified against the code, plus
`gosec` and `govulncheck`. Nothing was executed against a live deployment, so impacts are inferred
from code paths. Every finding carries a `file:line` reference to the tree above.

Severity scale: **Critical** (unauthenticated compromise of credentials or data integrity),
**High** (privilege escalation, cross-tenant impact, denial of service by one client, or a
shipped deployment that cannot run), **Medium**, **Low**, **Info**.

---

## 1. Executive summary

The fork inherits an upstream architecture in which **the scheduler trusts every network client**
and **the manager grants every self-registered account read access to every API group, including
plaintext personal access tokens**. The new data controller then layers a Kubernetes tenant
boundary on top of that trust model without enforcing one of its own: a namespace user who can
create a `Dataset` can exfiltrate the controller's manager token, read any Secret in their
namespace, direct the P2P fleet at internal addresses, and purge cached data that other tenants
depend on.

Three defects in the controller block it from running or observing at all:

- the shipped image cannot start under the shipped `securityContext` (read-only root filesystem
  plus an unconditional `MkdirAll` of `/usr/local/dragonfly/plugins`);
- Kubernetes Events are silently dropped because RBAC grants the core `events` resource while
  controller-runtime writes to `events.k8s.io`;
- a single malformed duration string in any `Dataset` can stall the informer for every namespace.

### Finding counts

| Area | Critical | High | Medium | Low/Info |
|---|---|---|---|---|
| Data controller (fork-specific) | 1 | 5 | 7 | 12 |
| Manager (upstream) | 1 | 4 | 9 | 6 |
| Scheduler and shared packages (upstream) | 2 | 6 | 8 | 4 |
| Supply chain, CI and deployment | 0 | 2 | 11 | 7 |

### Top ten actions, in order

1. **DC-C1** Stop sending the controller's global token to tenant-chosen endpoints; require
   `tokenSecretRef` whenever `managerRef.endpoint` is set, or move the endpoint into an
   admin-only, cluster-scoped object.
2. **DC-H5 / DC-L1** Make the controller runnable and observable: skip plugin-dir creation,
   add the `events.k8s.io` RBAC rule, add an image smoke test with the real `securityContext`.
3. **DC-H4 / DC-H2** Add CRD validation: duration patterns, `maxItems`/`maxLength`, URL scheme
   allow-list and private-range deny-list, CEL mutual-exclusion rules.
4. **DC-H1** Gate Secret references: opt-in label or Secret `type`, `SubjectAccessReview` in a
   validating webhook, and read Secrets through the API reader rather than a cluster-wide cache.
5. **DC-H3** Reference-count task IDs across live `Dataset`s before any `delete_task`.
6. **MGR-F1 / F5 / F2** Disable open signup by default, shrink `guest`, hash personal access
   tokens and never return them after creation, redact job `args` in storage and responses.
7. **MGR-F3** Authenticate `reset_password`, raise bcrypt cost, throttle sign-in, refuse the
   default root password.
8. **SCH-C1 / C2 / H1** Mandatory mTLS with certificate-to-host binding on the scheduler, seed
   membership from the manager only, no credential-header forwarding to self-declared seeds.
9. **SCH-H5 / MGR-F12** Bound inputs: `piece_count`, message size (2 GiB today), metric label
   cardinality, per-client rate limits, HTTP server timeouts and body caps.
10. **Supply chain** (§5) Pin everything, add provenance, and ship the controller with Helm,
    a NetworkPolicy, and a digest-pinned image.

---

## 2. Data controller (fork-specific)

Files: `datacontroller/**`, `cmd/datacontroller/**`, `deploy/data-controller/**`,
`build/images/datacontroller/Dockerfile`.

### DC-C1 Critical: controller token sent to a tenant-chosen endpoint

`datacontroller/controllers/dataset_controller.go:329-354` starts from the controller's own
endpoint and token, then lets `spec.managerRef.endpoint` replace the endpoint while leaving the
token in place when `tokenSecretRef` is absent. `datacontroller/dragonfly/client.go:298` then sends
`Authorization: Bearer <controller token>` to that endpoint. The CRD places no constraint on
`managerRef.endpoint`, and the same block lets the tenant set `insecureSkipTLSVerify`.

Impact: any subject with `create datasets` in any namespace obtains a manager personal access
token with the `job` scope (which, on the manager side, also covers cluster deletion, see
MGR-F5). The same field gives blind SSRF from the controller pod with the token attached;
non-2xx bodies are echoed into status and events (`client.go:316-317, 349-353`).

Fix: if `ref.Endpoint != ""` and `ref.TokenSecretRef == nil`, return an error. Better, remove
`endpoint` from the tenant API and add a cluster-scoped `ManagerConnection` (or a controller
allow-list) that admins own and `Dataset`s reference by name. Ignore tenant-level
`insecureSkipTLSVerify` unless an admin flag enables it. Add a unit test for the missing-token case
(the only existing `managerRef` test, `dataset_controller_test.go:681`, sets both fields).

### DC-H1 High: confused deputy over namespace Secrets

`deploy/data-controller/rbac/role.yaml:14-21` is a `ClusterRole` with `secrets: get, list, watch`,
bound cluster-wide. `datacontroller/controllers/operations.go:330-338` copies **every** key of the
referenced Secret into HTTP headers. Combined with `authSecretRef`, `credentialsSecretRef`,
`delegationTokenSecretRef` and `managerRef.tokenSecretRef`, a user with `create datasets` but no
`get secrets` can read any Secret in their namespace by referencing it and pointing the source at an
origin they control (the seed peers send the headers) or the manager at an endpoint they control
(DC-C1, the whole preheat request including credentials is POSTed).

Fix: require an opt-in label or a dedicated Secret `type`; refuse service-account-token and TLS
Secret types; add a validating webhook that performs a `SubjectAccessReview` for the creating user
against each referenced Secret; replace the cluster-wide grant with per-namespace `RoleBinding`s or
a namespace allow-list.

### DC-H2 High: source URLs are not validated

`operations.go:148-164` (`validateSource`) checks only that a URL is present.
`operations.go:203-207` forwards `url`/`urls` unchanged; the CRD has no `pattern` or `format`.
Peers (all of them under `scope: all_peers`) will fetch `http://169.254.169.254/...` or
`http://kubernetes.default.svc/...` and cache the result as a P2P task that any peer can read
back. Also unvalidated: `objectStorage.endpoint` plus a tenant-controlled
`objectStorage.insecureSkipVerify` (`dataset_types.go:155-165`), `distribution.ips`, and header
names (tenants can set `Host` or `Authorization` inline).

Fix: parse every URL; scheme allow-list from controller config; reject loopback, link-local,
RFC1918/CGNAT/ULA and metadata ranges unless allow-listed; reject hop-by-hop and sensitive header
names; add CEL `startsWith` and `maxLength` rules. Note DNS rebinding means the seed peer must
enforce the same check at fetch time.

### DC-H3 High: purge deletes tasks shared with other tenants

Task IDs derive only from URL, tag, application, piece length and filtered query params
(`operations.go:177-180`). `delete_task` jobs are issued per recorded task
(`operations.go:297-318`) to `schedulerClusterIDs`, which defaults to **all** clusters
(`dataset_types.go:216-218`). `reconcileDeletion` (`dataset_controller.go:526-548`) and
`expireSource` (`:504-507`) purge whatever is in `status.tasks` with no ownership or reference
count. Two teams declaring the same model URL, or a hostile tenant declaring a victim's URL and
then deleting with `deletionPolicy: Purge` or `ttl: 1s`, evicts the victim's cache fleet-wide.

Fix: before each `delete_task`, index live `Dataset`s cluster-wide and skip task IDs still
referenced (emit an event). Longer term, a cluster-scoped refcounted `DataTask` object. Let admins
bound `scope`, `schedulerClusterIDs` and purge rights per namespace.

### DC-H4 High: one bad duration can stall every namespace (confirm with envtest)

`ttl`, `refreshInterval`, `verifyInterval` and `timeout` are `type: string` with no pattern in
`deploy/data-controller/crds/*.yaml`. `metav1.Duration.UnmarshalJSON` fails on `ttl: "banana"`,
and a typed informer that cannot decode one object in a list or watch typically fails the whole
list. The finalizer then holds deletions too. Fix: `+kubebuilder:validation:Pattern` or CEL
`duration(self) > duration('0s')` on every `metav1.Duration`, plus minimum bounds (DC-M3).

### DC-H5 High (availability): the shipped image cannot start

`cmd/datacontroller/cmd/root.go:110` calls `dfpath.New`, which unconditionally runs
`os.MkdirAll(d.pluginDir, 0700)` (`pkg/dfpath/dfpath.go:82`) with
`DefaultPluginDir = "/usr/local/dragonfly/plugins"` (`pkg/dfpath/dfpath_linux.go:24`). The
Dockerfile creates only `/var/log/dragonfly` and runs as `USER 65532`; the Deployment sets
`readOnlyRootFilesystem: true` and mounts only the log dir. The process exits before the manager
starts. Fix: skip plugin-dir creation for this binary (or point it at an `emptyDir`), and add a CI
smoke test that runs the image with the Deployment's `securityContext`.

### Medium

- **DC-M1 Every Secret in the cluster is cached in memory.** `secretReader` uses the cached
  `mgr.GetClient()` (`datacontroller.go:86`); the first `Get` on a Secret starts a cluster-wide
  informer because `Cache.DefaultNamespaces` is set only when `controller.namespace` is non-empty
  (`datacontroller.go:76-78`). Use `mgr.GetAPIReader()` or `client.Options.Cache.DisableFor`,
  drop `list`/`watch`.
- **DC-M2 Plaintext transport and persisted credentials.** `configmap.yaml:13` ships
  `http://`; `config.go:351` and `client.go:215` accept it. Every request carries the bearer
  token, and preheat requests carry S3 keys, passwords, HDFS tokens and secret headers. The
  manager persists them in `jobs.args` and returns them on `GET /jobs` (MGR-F2). Default to
  `https`, add `manager.caFile`, require an explicit `allowInsecureHTTP`.
- **DC-M3 Manager load / DoS.** No lower bounds on intervals (`lifecycle.go:141,158,176`;
  `@every 1s` is a valid cron), no `maxItems` on `sources`/`urls`, `requeueNow()` at a fixed 1s
  (`dataset_controller.go:752-754, 1260-1261`) bypasses the workqueue limiter, serial `GetJob`
  polling with no reconcile deadline (`:660-682`), jobs re-fetched in `completeOperation`
  (`:846, :892`), `retryLimit` has no maximum, no client-side rate limiter.
- **DC-M4 Finalizer correctness.** Deletion hangs forever when the token Secret is gone
  (`:281-287`, `:601-603`); purge is skipped when lifecycle resolution fails (`:259-265`,
  `:592-594`); one invalid `DataLifecyclePolicy` selector fails every `Dataset` in the namespace
  (`lifecycle.go:57-60, 73-76`); the effective `deletionPolicy` is recomputed during deletion.
  Persist the resolved policy and endpoint in status at distribution time; skip invalid policies;
  add an admin `skip-purge` annotation.
- **DC-M5 Job creation is not idempotent.** `CreatePreheatJob` (`:717`) runs before the job ID
  is persisted; an optimistic-lock conflict (`:229`) or leader handover loses the ID and
  recreates the job. Write a `Creating` operation first and use an idempotency key in `BIO`.
- **DC-M6 Status task IDs are trusted for deletes.** `operations.go:298-304`. Anyone with
  `patch datasets/status` can delete arbitrary tasks fleet-wide. Recompute from spec or sign
  stored IDs; document that status write is privileged.
- **DC-M7 Unauthenticated metrics.** `datacontroller.go:69-74` binds `:8000` plain HTTP with no
  `SecureServing`/`FilterProvider`; no NetworkPolicy ships.

### Low / Info

- **DC-L1 Events are never written.** controller-runtime v0.23.1 records via `events.k8s.io/v1`
  (`manager.go:526` in the module); RBAC grants only core `events`. Every event is Forbidden and
  dropped; fake recorders in tests hide it. Add `apiGroups: [events.k8s.io]`.
- **DC-L2** `image: dragonflyoss/datacontroller:latest` (`deployment.yaml:29`); pin by digest.
- **DC-L3** `secret.yaml` placeholder is in `kustomization.yaml`; re-applying overwrites the real
  token. Use `secretGenerator` or document `kubectl create secret`.
- **DC-L4** Endpoint logged and put in errors unredacted (`datacontroller.go:116`,
  `client.go:212,216`); use `Redacted()`.
- **DC-L5** `r.clients` `sync.Map` keyed by `endpoint|token|...` never evicts
  (`dataset_controller.go:365-376`); plaintext tokens accumulate. Key by hash, use an LRU.
- **DC-L6** Manager/peer details and up to 512 bytes of manager error bodies are copied into
  tenant-visible status and events.
- **DC-L7** TTL computed from the latest `LastDistributionTime` across sources
  (`:1192-1203`), so an old source never expires; refresh outranks expire when both are due
  (`:488-495`).
- **DC-L8** Removed sources are rebuilt as `&DataSource{Name}` (`:521, :544`); if `task.ID` is
  empty the computed ID omits tag/application/pieceLength and targets the wrong task.
- **DC-L9** Policy reconciler watches only its own generation; a priority change on policy A does
  not recompute policy B.
- **DC-L10** Leader-election Role grants `leases: delete` unnecessarily; no PodDisruptionBudget.
- **DC-L11** `retryLimit` default is 3 in the CRD but 0 in code when nil (`:970-973`).
- **Info** No path traversal found; token loading via file/env is sound; pod `securityContext`
  is good (`runAsNonRoot`, uid 65532, seccomp, `drop: ALL`, read-only root).

### Test gaps on security paths

No tests for: `managerRef.endpoint` without `tokenSecretRef`; URL/scheme/header validation;
disallowed Secret types; two `Dataset`s sharing a task with one purged; forged status IDs;
deletion with the manager unavailable, token missing, or a sibling policy invalid; status-patch
conflicts; event RBAC with a real recorder; running the image under the shipped
`securityContext`. There is no envtest, no kind e2e, and no CRD drift check in CI.

---

## 3. Manager (upstream code)

Files: `manager/**`, `internal/job/**`, `pkg/oci/**`, `pkg/net/http/**`.

### MGR-F1 Critical: open signup + `guest` reads every API group

`manager/router/router.go:112` registers `POST /api/v1/users/signup` with no middleware;
`manager/service/user.go:133` assigns `guest`. `manager/permission/rbac/rbac.go:133` grants
`guest` `read` on **every** API group, including `personal-access-tokens`, whose model exposes the
secret with `json:"token"` (`manager/models/personal_access_token.go:33`). Chain: sign up, sign in,
`GET /api/v1/personal-access-tokens`, then use any stolen token on `/oapi/v1/*`, which bypasses
Casbin entirely. The same account can dump users (email, phone), audits, configs and all job
results.

### MGR-F2 High: credentials persisted in `jobs.args` and returned

`manager/service/job.go:134` stores `PreheatArgs` (username, password, headers, object-storage
and HDFS credentials) via `StructToMap` into `models.Job.Args` (`manager/models/job.go:27`), and
`GET /jobs` returns it unredacted. Redact or encrypt before persisting; return a DTO.

### MGR-F3 High: password handling

`bcrypt.MinCost` (cost 4) in `rbac.go:154`, `service/user.go:98,113`.
`POST /users/:id/reset_password` (`router.go:116`) has **no** JWT or RBAC middleware and no
throttle, so it is an unauthenticated password oracle against any user ID (root is 1). No lockout
or rate limit on `signin`. `DefaultRootPassword = "dragonfly"` only logs a warning. Password max
length 20 blocks passphrases.

### MGR-F4 High: manifest fetch disables TLS verification; constrained SSRF

`pkg/oci/image.go:41` hard-codes `InsecureSkipVerify: true` and ignores the configured CA;
registry credentials are sent over it. `SafeDialer` (`pkg/net/http/http.go:52-79`) blocks
loopback and link-local but `IsGlobalUnicast()` admits RFC1918, CGNAT (100.64/10, including
Alibaba metadata 100.100.100.200) and IPv6 ULA. The manifest regex
`^(.*)://(.*)/v2/(.*)/manifests/(.*)` (`pkg/oci/reference.go`) imposes no host or scheme limit.

### MGR-F5 High: personal access tokens

Stored plaintext and matched with `Where("token = ?")`
(`manager/middlewares/personal_access_token.go`); returned on every `GET`; accepted from the
`?access_token=` query string (logged by gin-zap); empty scopes mean **all** permissions;
default scopes include `clusters`, which allows `DELETE /oapi/v1/clusters/:id`; PATs skip Casbin
so a `guest`-owned token acts as admin on jobs and clusters; `CreatePersonalAccessTokenRequest.UserID`
and `CreateJobRequest.UserID` are client-supplied. Hash with a prefix, show once, add read/write
scopes, intersect with the owner's RBAC, derive `user_id` from the caller.

### Medium

- **MGR-F6** CORS reflects any `Origin` with `Allow-Credentials: true` and reflects requested
  headers (`manager/middlewares/cors.go:35-45`); JWT cookie is not `HttpOnly`, no `Secure`, no
  `SameSite`; JWT accepted from `?token=`.
- **MGR-F7** No JWT revocation; `UserStateDisabled` is never enforced; 14-day timeout plus 7-day
  refresh; Casbin policy loaded once with no watcher, so grants and revocations do not propagate
  across replicas.
- **MGR-F8** OAuth `state` generated but never verified (login CSRF); users keyed by mutable
  display `Name`; duplicate-name path returns an unsaved user with `ID == 0`; nil-pointer panic
  on private GitHub email; over-broad `user public_repo` scopes.
- **MGR-F9** gRPC TLS optional; with mTLS any CA-signed cert can call every RPC (no identity to
  argument binding); certs loaded once; reflection always on; `MaxRecvMsgSize(math.MaxInt32)`.
- **MGR-F10** `/metrics` registered unauthenticated on the REST engine by `go-gin-prometheus`;
  URL label uses the raw path, so unauthenticated 404 spam creates unbounded series.
- **MGR-F11** Audit log: no IP, user agent or request ID; `Detail` never populated; drop on full
  channel (1000); a failing batch is retried forever and `Path` (attacker-controlled, up to any
  length) into a `varchar(1024)` column can poison it; purgeable via a `gc` job by anyone with job
  write; 7-day TTL; no tamper evidence or export.
- **MGR-F12** Job rate limiter returns `true` for an empty `scheduler_cluster_ids`
  (`internal/ratelimiter/job_ratelimiter.go`) while the job still targets all clusters;
  `per_page` up to 10,000,000; `args.timeout` unbounded; `http.Server` has no timeouts or
  `MaxHeaderBytes`; no `MaxBytesReader`.
- **MGR-F13** Secrets in logs: `job/preheat.go:206`, `job/task.go:120,194` log signatures with
  `%#v` at Info (resolved `Authorization` tokens, object-storage keys); gin-zap logs raw query
  strings.
- **MGR-F14** Machinery broker/backend ignores Redis TLS (`internal/job/job.go` builds `redis://`
  URLs); job payloads with credentials travel and rest in cleartext in Redis. Anyone who can write
  to the broker DB (including via the raw TCP proxy on `:65100`) can inject jobs with arbitrary
  URLs and credentials; payloads are unsigned.

### Low / Info

Env override binds every config key including `MANAGER_AUTH_JWT_KEY` and DB passwords, non-strict
decoding; `X-Server-IP` header discloses internal addresses; `SetTrustedProxies` never called so
`ClientIP()` is spoofable; swagger UI always public; gin stays in debug mode unless `logLevel`
is exactly `info`; Redis cache keys use `-` delimiters from unauthenticated fields (collision);
searcher plugin loaded from the plugin dir at startup. No SQL injection or command execution
found; all GORM calls use struct conditions or placeholders.

---

## 4. Scheduler and shared packages (upstream code)

Files: `scheduler/**`, `pkg/rpc/**`, `pkg/redis/**`, `pkg/idgen/**`, `pkg/digest/**`,
`internal/dfplugin/**`, `cmd/dependency/**`.

### SCH-C1 Critical: unauthenticated gRPC, identity-blind mTLS

`scheduler/scheduler.go:181` defaults to `rpc.NewInsecureCredentials()`; TLS is commented out in
`deploy/docker-compose/template/scheduler.template.yaml:14-21`. With TLS,
`pkg/rpc/credential.go:49-53` only verifies the chain. A grep for `peer.FromContext`, `AuthInfo`
and `PeerCertificates` across `scheduler/`, `pkg/`, `internal/` returns nothing; the interceptor
chain (`pkg/rpc/scheduler/server/server.go:62-80`) has no auth. Any client reaching port 8002 can
call `AnnounceHost`, `AnnouncePeer`, `DeleteHost`, `DeleteTask`, `DeletePeer`, `ListHosts`,
`PreheatImage`, `PreheatFile`, and all `*Persistent*` RPCs.

### SCH-C2 Critical: self-declared seed peers receive other users' requests and headers

`service_v2.go:548-560, 667` accepts `Host.Type` from the client (proto bound `lte = 3`);
`scheduler/resource/standard/host_manager.go:142-143` puts type `1` into the `seeds` map;
`seed_peer.go:355-380` builds the seed hash ring from `LoadAllSeeds()` plus a health check the
attacker can trivially pass. `service_v2.go:1360-1376, 1894-1960` forward the victim's `Download`
(URL, `RequestHeader`, digest) to `hashring.Get(taskID)`. Result: credential theft for
back-to-source requests and poisoned content served to the fleet. Fix: seed membership from the
manager (dynconfig) or a certificate attribute only; strip credential headers before forwarding to
anything that is not a configured seed.

### High

- **SCH-H1 Host ID squatting.** `pkg/idgen/host_id.go:22-27` is `ip-hostname`; for an existing
  ID `AnnounceHost` refreshes ports and type but never IP or hostname and never checks ownership
  (`service_v2.go:664-667`). First announce wins; hosts can be taken over or their ports redirected.
- **SCH-H2 Task metadata poisoning.** `handleResource` (`service_v2.go:1839-1892`) trusts the
  client's task ID, never recomputes it via `idgen.TaskIDV2`, and overwrites URL, filtered params
  and headers for an existing task (`:1869-1871`). Piece reports (`:1771`) overwrite by number from
  any caller supplying any peer ID. The scheduler never verifies digests.
- **SCH-H3 Credential disclosure.** `StatTask` returns `RequestHeader: task.Header`
  (`service_v2.go:473`), `StatPeer` likewise (`:329`), `ListHosts` dumps the fleet; request headers
  are logged at Info with `%#v` (`:147-148, :549`; `peer.go:499`).
- **SCH-H4 Unauthenticated mutating and admin RPCs.** `DeletePeer` (`:446`), `DeleteTask`
  (`:519`), `DeleteHost` (`:1210`), `DeletePersistent*` (`:3206, :4416`), `Preheat*`/`Stat*`
  (`:4475-5020`) with caller-supplied URL, credentials, `Scope=all_peers`, `InsecureSkipVerify`.
- **SCH-H5 Resource exhaustion from one client.** `bitset.New(uint(req.GetPieceCount()))`
  (`:2920, :4126`) with unbounded `uint32` (about 512 MiB per call); `FinishedPieces.Set(Number)`
  grows to any `uint32`; unbounded hosts, peers, tasks (24 h peer TTL); client-supplied OS,
  platform, kernel and build strings as Prometheus labels (`scheduler_server_v2.go:131-132`,
  `metrics.go:211-217`); `MaxRecvMsgSize(math.MaxInt32)` (`server.go:60`); a single global
  limiter (4000 rps) that one client can drain; `io.ReadAll` without a limit on peer-supplied
  bodies (`peer.go:514`).
- **SCH-H6 SSRF via announced addresses.** Host IP is validated only as `string.ip`; the scheduler
  dials `IP:DownloadPort` plaintext with the default dialer at eight sites (`:2852, :3055, :3096,
  :3232, :4058, :4265, :4303, :4442`). `SafeDialer` elsewhere admits RFC1918 and ULA.

### Medium

- **SCH-M1** Native plugin loading (`internal/dfplugin/dfplugin.go:54-55`) with no ownership,
  mode or signature check; `pkg/dfpath/dfpath.go:82` sets `0700` only on creation; env override
  can redirect `server.pluginDir`.
- **SCH-M2** Redis key injection: keys built with `fmt.Sprintf` and `:` from IDs validated only
  as `min_len = 1` (`pkg/redis/redis.go:155,164,185,200`); a crafted task ID aliases another
  task's peer-set key (WRONGTYPE or deletion).
- **SCH-M3** Manager, seed-peer, peer and Redis TLS all optional with plaintext fallback
  (`scheduler.go:343-360`); `insecureSkipVerify` accepted.
- **SCH-M4** statsview/pprof binds `0.0.0.0` when enabled (`cmd/dependency/dependency.go:185-200`);
  `/metrics` on `:8000` unauthenticated; `reflection.Register` always on (`server.go:99`); OTLP
  exporter uses insecure credentials with auth headers in cleartext; 100 percent sampling.
- **SCH-M5** `pkg/oci/image.go:41` `InsecureSkipVerify: true` for registry fetches, ignoring the
  request's `CertificateChain`.
- **SCH-M6** `pkg/digest/digest.go:25-37` accepts `md5`, `sha1`, `crc32` for integrity;
  `SHA256FromStrings` (`:140-154`) and `TaskIDV2ByURLBased` concatenate inputs without delimiters
  (chosen-collision across different tuples); blob-digest task IDs omit the algorithm and registry.
- **SCH-M7** Persistent and persistent-cache tasks can be created or overwritten by any caller in
  `Pending`/`Failed` (`:2880-2933, :4086-4140`); no owner concept.
- **SCH-M8** Peer identity not bound to the stream (`:121-290`); a hostile stream registering a
  victim's peer ID removes the victim's real stream on exit (`:152` defer); any peer can report
  `DownloadPieceFailed` against a victim to poison `BlockParents`.

### Low / Info

Misleading "skip hostname verification" comment in `pkg/rpc/credential.go:73-97` (standard
verification still runs); unused `InsecureSkipVerify` on an `http://`-only path (`peer.go:259`);
no explicit TLS `MinVersion`; log files `0644` (`internal/dflog/logger.go:410,423`) combined with
SCH-H3. Clean: no `os/exec`; `math/rand` only for jitter; peer IDs are `uuid.NewString`;
local dynconfig written `0600`.

---

## 5. Supply chain, CI and deployment

What is already good: every third-party action is pinned to a full commit SHA (the two exceptions
are the documented tag pin for `slsa-github-generator@v2.1.0` and a digest-pinned markdownlint
image); `pull_request_target` appears only in `auto-assign.yml` and never checks out PR code; no
`curl | sh`, no `${{ github.event.* }}` in `run:` blocks; CodeQL, Scorecards, SLSA3 provenance
for releases, buildx provenance, SBOM generation and cosign signing for images all exist; base
images are digest-pinned; the Go toolchain in the base image is checksum-verified; no real private
keys are committed (the three `testdata` cert and key files are 4-byte placeholders).

| # | Sev | Finding | Location | Fix |
|---|---|---|---|---|
| SC-1 | High | `pull_request` jobs run PR code with `sudo` on persistent self-hosted runners and receive `DOCKER_USERNAME`/`DOCKER_PASSWORD` in the unit-test step, which `make test` does not use. Severity depends on repo visibility and fork-PR approval settings, which were not visible. | `.github/workflows/ci.yml:20,34-48,60`; `e2e*.yml` | GitHub-hosted or ephemeral runners for `pull_request`; require approval for fork PRs; drop the Docker secrets from the test step |
| SC-2 | High | 20 known vulnerabilities in 6 modules reachable by call graph (see §6) | `go.mod` | Bump `rabbitmq/amqp091-go` to v1.14.0, `golang.org/x/net` to v0.60.0, `grpc` to v1.83.2, `klauspost/compress` to v1.20.2, `otlptracehttp` to v1.45.0; plan the replacement of `docker/docker v25.0.16+incompatible`, which has no fixed version |
| SC-3 | Medium | `permissions: write-all` on the job holding signing and registry secrets | `.github/workflows/docker.yml:16` | `contents: read`, `packages: write`, `id-token: write`, `security-events: write` |
| SC-4 | Medium | Images are pushed before Trivy runs and Trivy has no `exit-code`, so CRITICAL/HIGH findings never fail the job | `docker.yml:106-116` | Build with `load`, scan with `exit-code: '1'`, push only on pass |
| SC-5 | Medium | Scheduler and manager runtime images have no `USER`; only the datacontroller image is non-root | `build/images/scheduler/Dockerfile`, `build/images/manager/Dockerfile` | Add `USER 65532:65532` and chown the writable dirs |
| SC-6 | Medium | `hack/docker-build.sh` defaults `BASE_IMAGE=alpine:3.23` (mutable tag) and always passes it, overriding the digest pin in every Dockerfile for local builds | `hack/docker-build.sh:13,38` | Default to the digest-pinned value |
| SC-7 | Medium | docker-compose binds every port to 0.0.0.0, scheduler gRPC (8002), peer upload (4000) and proxy (4001) are plaintext, mTLS is commented out, no `cap_drop`/`read_only`/`no-new-privileges` | `deploy/docker-compose/**` | Internal network or loopback binds, enable TLS, add hardening options |
| SC-8 | Medium | `:latest` images with `IfNotPresent` | `docker-compose.yaml:3,24,40`; `deploy/data-controller/deployment.yaml:29` | Pin by digest |
| SC-9 | Medium | Placeholder Secret listed in kustomization; re-apply overwrites the live token | `deploy/data-controller/kustomization.yaml:12` | Remove; use `secretGenerator`, ExternalSecrets or SealedSecrets |
| SC-10 | Medium | Release: `goreleaser-action` at `version: latest`, `--skip=validate`, no `signs:`/`sboms:` blocks, no `-trimpath`, build time embedded (non-reproducible) | `.github/workflows/release.yml:36,38`; `.goreleaser.yml:85-86`; `hack/build.sh:15,22` | Pin goreleaser, drop skip, add cosign and SBOM, `-trimpath`, `SOURCE_DATE_EPOCH` |
| SC-11 | Medium | `grpc_health_probe` downloaded into runtime images with no checksum verification | `build/images/manager/Dockerfile:34-43`; `build/images/scheduler/Dockerfile:19-28` | Verify SHA256 or build from source in a pinned stage |
| SC-12 | Medium | Lint config omits `gosec`, `bodyclose`, `noctx`; `gocyclo` threshold 100; `std-error-handling` preset suppresses errcheck on Close/Write; the `// nolint:gosec` in `client.go:228` is inert | `.golangci.yml:4-22` | Enable the linters; scope the preset |
| SC-13 | Medium | Dependabot covers `gomod` and `github-actions` only; Dockerfile digests never get bumped | `.github/dependabot.yml` | Add `docker` ecosystem and grouping |
| SC-14 | Low | No NetworkPolicy or PodDisruptionBudget for the two-replica controller | `deploy/data-controller/` | Add both |
| SC-15 | Low | nerdctl tarball installed without checksum; `nydus-latest` image tag | `.github/workflows/e2e-nydus.yml:118-123` | Verify and pin |
| SC-16 | Low | `persist-credentials: false` only in `scorecards.yml` | all other checkouts | Set it where no push is needed |
| SC-17 | Low | `SECURITY-INSIGHTS.yml` last updated 2025-07-15; claims regular assessments and an automated signed pipeline that release artifacts do not match | `SECURITY-INSIGHTS.yml:6-7,53-61` | Refresh and trim unsupported claims |
| SC-18 | Low | `.gitignore` lacks `*.key`, `*.pem`, `*.crt`, `.env`, `secret*.yaml` | `.gitignore` | Add with testdata negations |
| SC-19 | Info | `AKIAEXAMPLE`/`EXAMPLESECRET` sample values trip secret scanners | `samples/dataset-object-storage.yaml:10-11` | Use `REPLACE_ME` |
| SC-20 | Info | Cosign is key-based (`COSIGN_KEY` secret) and signs by tag; SBOMs are generated but not attached as attestations | `docker.yml:125-148` | Keyless signing, verify by digest, attest SBOM |

Not audited: `deploy/helm-charts` is a submodule and is not checked out in this tree. Repo
settings (branch protection, fork-PR approval, runner registration) were not visible.

Missing supply-chain features: dependency-review action on PRs; govulncheck and Trivy as blocking
PR gates; keyless cosign with digest verification; SBOM attestations; signed release checksums;
pinned goreleaser; Dependabot docker ecosystem; NetworkPolicy and PDB for the controller; a
controller image smoke test under the shipped `securityContext`.

---

## 6. Scanner results

### govulncheck (source mode, `./cmd/...`, run 2026-11-07)

"Your code is affected by 20 vulnerabilities from 6 modules." No standard-library entries.

| Module | Installed | Fixed in | IDs |
|---|---|---|---|
| `github.com/rabbitmq/amqp091-go` | v1.9.0 | v1.14.0 (v1.13.0 for most) | GO-2026-6700, 6503, 6502, 6500, 6498, 6496, 6494, 6492, 6491, 6372 |
| `golang.org/x/net` | v0.57.0 | v0.60.0 | GO-2026-6617, 6612, 6611, 6610, 6603 (HTTP/2 server crash, memory exhaustion, CPU, flow control) |
| `google.golang.org/grpc` | v1.83.1 | v1.83.2 | GO-2026-6443 |
| `github.com/klauspost/compress` | v1.19.1 | v1.20.2 | GO-2026-6865 |
| `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp` | v1.44.0 | v1.45.0 | GO-2026-6505 (endpoint URL leak in logs) |
| `github.com/docker/docker` | v25.0.16+incompatible | none | GO-2026-4887, GO-2026-4883 (reached via `pkg/oci/auth.go` registry init; exploitability unverified) |

The `x/net` HTTP/2 entries matter most: the manager REST server and every gRPC listener are
network-facing and unauthenticated by default. `amqp091-go` is pulled in by machinery; the Redis
broker is used, so the AMQP code is likely unreachable at runtime but is still linked.

### gosec v2.30.0 (`./...`, tests and generated code excluded)

| Rule | Count | Notes |
|---|---|---|
| G115 integer overflow conversions | 68 | Mostly `uint64`/`int` to `uint32`/`int32`. Review the ones on network inputs: `scheduler/service/service_v2.go:1879`, `service_v1.go:949`, `datacontroller/controllers/operations.go:496` (`int32(len(result.Peers))`) |
| G118 unbounded size (slice/alloc from input) | 25 | Overlaps SCH-H5 |
| G304 file inclusion via variable | 7 | All config-controlled paths (`pkg/rpc/credential.go:39,63`, `cmd/dependency/dependency.go:323,334`, `internal/dflog/logger.go:410,423`, `pkg/types/types.go:71`); acceptable given env override caveats |
| G402 `InsecureSkipVerify: true` | 3 | `pkg/oci/image.go:41` (hard-coded, MGR-F4/SCH-M5), `scheduler/resource/standard/peer.go:259` (dead on an `http://` path), `datacontroller/dragonfly/client.go:228` (config-gated; the `// nolint:gosec` is inert because gosec is not in `.golangci.yml`) |
| G112 no `ReadHeaderTimeout` | 3 | `manager/manager.go:164-167` (REST), `manager/metrics/metrics.go:78-81`, `scheduler/metrics/metrics.go:699-702` (MGR-F12) |
| G123 `VerifyPeerCertificate` without `VerifyConnection`, session resumption enabled | 1 | `pkg/rpc/credential.go:76`: resumed TLS sessions skip the custom check. Set `SessionTicketsDisabled` or add `VerifyConnection` |
| G302 file mode above 0600 | 2 | `internal/dflog/logger.go:410,423` (0644 log files; combine with SCH-H3 header logging) |
| G404 `math/rand` | 2 | `pkg/time/delay.go:38,59`, jitter only; not security-relevant |
| G101 hard-coded credential names | 2 | Constant names for Secret keys; false positives |
| G104 unhandled error | 2 | Minor |

Full outputs were kept outside the repository during the review; re-run with
`go run golang.org/x/vuln/cmd/govulncheck@latest ./cmd/...` and
`go run github.com/securego/gosec/v2/cmd/gosec@latest -exclude-dir=test -exclude-generated ./...`.

---

## 7. Missing features

### 7.1 Data controller

| Pri | Gap | Proposal | Effort |
|---|---|---|---|
| P0 | No CI drift check, envtest, or kind e2e for the controller; e2e suites never deploy it | `make generate-datacontroller && git diff --exit-code`; envtest for CRD schema and finalizer flow; kind e2e applying `samples/dataset-files.yaml` | M |
| P0 | Shared-task purge protection (DC-H3) | Cluster-wide index of `status.tasks[].id` owners; skip and event on conflict; later a refcounted `DataTask` | M |
| P0 | No admission validation or defaulting; no CEL rules | CEL `XValidation` for mutual exclusion (`refreshInterval` vs `refreshSchedule`), type vs url/platform, duration patterns, `maxItems`/`maxLength`; webhook for cron parsing, Secret SAR, selector overlap | S/M |
| P1 | Conditions lack `Progressing`/`Stalled`; no policy conditions | kstatus-compatible set, per-condition `observedGeneration`, `Accepted`/`Conflicted` on policies | S |
| P1 | Two counters only (`metrics.go`) | Histograms for operation and manager-request latency; gauges for datasets by phase, in-flight ops, next refresh; failure counters by reason; bytes | S/M |
| P1 | No progress or size in status | `tasks[].contentLength`, `status.progress{totalTasks,completedTasks,totalBytes}`, job `startedAt`/`finishedAt` | M |
| P1 | `deploy/helm-charts/` is empty; kustomize only; no ServiceMonitor, PDB, NetworkPolicy, HPA; `:latest` image | Standalone chart with CRDs, values for endpoint/token, ServiceMonitor, PDB, NetworkPolicy, digest-pinned image | M |
| P1 | Source types are `file` and `image` only | `huggingface` (repo, revision, allowPatterns, token) expanded controller-side; `oci-artifact` | M/L |
| P2 | Multi-tenancy: no quotas, no allowed-manager list, tenant-settable endpoint | Cluster-scoped `DataTenantPolicy`: allowed endpoints, scopes, clusters, max sources, per-namespace job and byte budgets; per-namespace tokens | M |
| P2 | Scheduler-cluster targeting by numeric ID only; no `seedPeerClusterIDs`; no CA/client-cert for manager or origin | Resolve cluster names via manager API; `caBundleSecretRef` replacing `insecureSkipTLSVerify` | M |
| P2 | Retention is TTL-only | `retention{maxVersions,maxTotalBytes,evictionOrder}`; needs size reporting | L |
| P2 | Retry knobs thin; failed sources park forever | Per-dataset backoff, `retry-now` annotation, cool-off re-attempt | S |
| P2 | Versioning: v1alpha1 only, no conversion strategy | Publish stability policy; hub/spoke conversion before v1beta1 | M |
| P3 | No `dependsOn`/priority, `nodeSelector` placement, dry-run, multi-manager, kubectl plugin, API reference docs, runbooks, `tools.go` pin for controller-gen | As listed | S–L |

### 7.2 Manager

MFA/TOTP/WebAuthn; OIDC/SAML/LDAP with group-to-role mapping; account lockout, password policy,
forced first-login change, enforced disable, session listing and revocation; hashed PATs with
one-time reveal, rotation, last-used, read-only scopes, IP binding, owner-RBAC intersection;
encryption at rest for OAuth secrets and job args, Vault/KMS or `*File` config options, log
redaction; resource- and cluster-level authorization, ownership checks, approval step for
`delete_task` and `gc`, policy-as-code for Casbin; audit with IP, request ID, diffs, durable queue,
hash chain, SIEM export, separate purge permission; signed job payloads, broker TLS, egress
allow-list and private-range deny-list, honour configured CA, bounded `timeout` and concurrency;
HTTP timeouts and body caps, per-IP/per-user auth rate limits, security headers, CORS allow-list,
authenticated `/metrics` and swagger, trusted-proxy config, cert hot-reload and revocation.

### 7.3 Scheduler and data plane

Peer identity and attestation (certificate-to-host binding, token, or SPIFFE); admin versus peer
RPC roles; manager-signed seed membership; signed piece or task manifests and server-side digest
verification; strong digest policy (reject md5/sha1/crc32); server-side task ID recomputation;
encryption in transit by default (peers, seeds, manager, Redis, tracing) and encryption at rest
for Redis state; credential redaction and no header forwarding to untrusted hosts; per-tenant
key-space isolation; per-client quotas and rate limits with caps on hosts, peers, tasks, pieces
and message size; URL scheme and host admission policy for back-to-source; peer reputation and
anomaly detection; plugin signature verification; authenticated metrics, pprof and reflection off
by default.

### 7.4 Whole project, enterprise and HPC

Content signature verification (cosign/notation) for file and image sources; end-to-end OTel
tracing from reconcile through manager job to scheduler; capacity and quota surfaces (peer disk
watermarks); manager DB backup and restore runbook; HA Redis guidance; CRD and controller upgrade
tests in `e2e-compatibility.yml`; FIPS 140-3 build variant; offline/air-gapped bundle;
ready-made `ValidatingAdmissionPolicy` samples; audit export sink.

---

## 8. Suggested sequencing

Sprint 1 (unblock and contain): DC-H5, DC-L1, DC-C1, DC-H4, CI drift check and image smoke test.
Sprint 2 (tenant boundary): DC-H1, DC-H2, DC-H3, DC-M1, DC-M3 minimums, CEL rules.
Sprint 3 (manager hygiene): MGR-F1, F2, F3, F5, F6, F12, F13.
Sprint 4 (data plane trust): SCH-C1, C2, H1, H3, H4, H5 bounds; manager F9/F14.
Sprint 5 (packaging and features): Helm chart, metrics, conditions, progress, Hugging Face source.

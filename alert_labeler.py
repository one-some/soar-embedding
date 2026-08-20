import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Alert:
    id: str
    timestamp: datetime
    source: str
    raw_data: Dict

    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    hostnames: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    users: Set[str] = field(default_factory=set)
    processes: Set[str] = field(default_factory=set)
    files: Set[str] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)

    rule_groups: List[str] = field(default_factory=list)
    rule_id: Optional[str] = None
    compliance_tags: Dict[str, List[str]] = field(default_factory=dict)

    signature: Optional[str] = None
    category: Optional[str] = None
    severity: int = 0
    full_log: Optional[str] = None

    label: str = "UNLABELED"
    confidence: float = 0.0
    label_method: str = ""
    attack_phase: Optional[str] = None


class EnhancedEntityExtractor:
    ATTACK_PATTERNS = {
        "network_scans": [
            r"\b(?:nmap|masscan|zmap)\b",
            r"SYN.*scan",
            r"port.*scan",
            r"network.*reconnaissance",
            r"ICMP.*sweep",
        ],
        "service_scans": [
            r"\b(?:nikto|nmap.*-sV|whatweb)\b",
            r"service.*detection",
            r"version.*scan",
            r"banner.*grab",
            r"web server 400 error code",
            r"forbidden file or directory",
            r"suspicious url access",
        ],
        "wpscan": [
            r"\bwpscan\b",
            r"wordpress.*scan",
            r"wp-admin",
            r"wp-login",
            r"xmlrpc",
            r"web server 400 error code",
            r"forbidden directory index",
            r"forbidden file or directory",
        ],
        "dirb": [
            r"\b(?:dirb|dirbuster|gobuster|ffuf)\b",
            r"directory.*brute",
            r"\.git\b",
            r"\.env\b",
            r"admin\.php",
            r"forbidden.*directory",
            r"web server 400 error code",
            r"suspicious url access",
            r"common web attack",
            r"web server 500 error code",
        ],
        "webshell": [
            r"\b(?:shell|webshell|backdoor|c99|r57)\b",
            r"system\(",
            r"exec\(",
            r"passthru\(",
            r"\.php.*cmd=",
            r"base64.*decode",
        ],
        "cracking": [
            r"\b(?:john|hashcat|hydra|medusa)\b",
            r"password.*crack",
            r"brute.*force",
            r"authentication.*fail.*\d{3,}",
            r"failed.*login.*attempts",
            r"cms \(wordpress or joomla\) login attempt",
        ],
        "reverse_shell": [
            r"reverse.*shell",
            r"/bin/(?:bash|sh).*-i",
            r"nc.*-e.*sh",
            r"/dev/tcp/",
            r"python.*pty\.spawn",
        ],
        "privilege_escalation": [
            r"privilege.*escalation",
            r"sudo.*exploit",
            r"setuid",
            r"SUID",
            r"capability.*exploit",
            r"/etc/passwd",
            r"/etc/shadow",
        ],
        "service_stop": [
            r"service.*stop",
            r"systemctl.*stop",
            r"kill.*-9",
            r"daemon.*terminate",
            r"process.*killed",
        ],
        "dnsteal": [
            r"dns.*tunnel",
            r"dns.*exfil",
            r"high entropy.*dns",
            r"suspicious.*subdomain",
            r"long.*dns.*query",
            r"base64.*\.[a-z]+",
        ],
    }

    COMPILED_PATTERNS = {
        phase: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for phase, patterns in ATTACK_PATTERNS.items()
    }

    def __init__(self):
        self.ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self.domain_pattern = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
        self.user_pattern = re.compile(
            r'(?:user|acct|username|uid|account)[=:\s]+["\']?([a-zA-Z0-9_-]+)["\']?',
            re.I,
        )
        self.process_pattern = re.compile(
            r'(?:exe|cmd|command|process)[=:\s]+["\']?([^\s"\']+)["\']?', re.I
        )
        self.file_pattern = re.compile(
            r'(?:file|path)[=:\s]+["\']?([^\s"\']+)["\']?', re.I
        )
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.I)

    def extract_wazuh(self, alert_data: Dict) -> Alert:
        alert_id = alert_data.get("id", str(hash(json.dumps(alert_data))))
        timestamp_str = alert_data.get("@timestamp") or alert_data.get("data", {}).get(
            "timestamp"
        )
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

        alert = Alert(
            id=alert_id, timestamp=timestamp, source="wazuh", raw_data=alert_data
        )

        data = alert_data.get("data", {})
        alert.src_ip = data.get("src_ip")
        alert.dst_ip = data.get("dest_ip")
        alert.src_port = data.get("src_port")
        alert.dst_port = data.get("dest_port")

        dns_data = data.get("dns", {})
        if "query" in dns_data:
            for query in dns_data["query"]:
                if "rrname" in query:
                    alert.domains.add(query["rrname"])

        http_data = data.get("http", {})
        if http_data:
            if "url" in http_data:
                alert.urls.add(http_data["url"])
            if "hostname" in http_data:
                alert.hostnames.add(http_data["hostname"])

        rule = alert_data.get("rule", {})
        alert.rule_id = rule.get("id")
        alert.rule_groups = rule.get("groups", [])
        alert.signature = rule.get("description")
        alert.severity = int(rule.get("level", 0))

        for tag in ["pci_dss", "nist_800_53", "gdpr", "tsc", "gpg13", "mitre"]:
            if tag in rule:
                alert.compliance_tags[tag] = rule[tag]

        alert_info = data.get("alert", {})
        if alert_info:
            alert.category = alert_info.get("category")
            if not alert.signature:
                alert.signature = alert_info.get("signature")
            if not alert.severity and "severity" in alert_info:
                alert.severity = int(alert_info["severity"])

        alert.full_log = alert_data.get("full_log", "")
        if alert.full_log:
            self._extract_from_full_log(alert, alert.full_log)

        predecoder = alert_data.get("predecoder", {})
        if "hostname" in predecoder:
            alert.hostnames.add(predecoder["hostname"])
        if "program_name" in predecoder:
            alert.processes.add(predecoder["program_name"])

        return alert

    def extract_aminer(self, alert_data: Dict) -> Alert:
        alert_id = str(hash(json.dumps(alert_data)))

        log_data = alert_data.get("LogData", {})
        timestamps = log_data.get("Timestamps", [])
        timestamp = (
            datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
            if timestamps
            else datetime.now(timezone.utc)
        )

        alert = Alert(
            id=alert_id, timestamp=timestamp, source="aminer", raw_data=alert_data
        )

        analysis = alert_data.get("AnalysisComponent", {})
        alert.signature = analysis.get("AnalysisComponentName", "")
        alert.category = analysis.get("Message", "")

        for raw_log in log_data.get("RawLogData", []):
            alert.full_log = raw_log
            self._extract_from_full_log(alert, raw_log)

        if any(
            word in alert.signature.lower() for word in ["anomaly", "new", "suspicious"]
        ):
            alert.severity = 2
        else:
            alert.severity = 1

        return alert

    def _extract_from_full_log(self, alert: Alert, log_text: str):
        ips = self.ip_pattern.findall(log_text)
        if ips:
            if not alert.src_ip and len(ips) >= 1:
                alert.src_ip = ips[0]
            if not alert.dst_ip and len(ips) >= 2:
                alert.dst_ip = ips[1]

        for domain in self.domain_pattern.findall(log_text):
            if not re.match(r"\d+\.\d+", domain) and len(domain) > 3:
                alert.domains.add(domain)
                parts = domain.split(".")
                if len(parts) >= 2:
                    alert.hostnames.add(".".join(parts[-2:]))

        for user in self.user_pattern.findall(log_text):
            if user and user != "?":
                alert.users.add(user)

        for proc in self.process_pattern.findall(log_text):
            if proc:
                alert.processes.add(proc)

        for file_path in self.file_pattern.findall(log_text):
            if file_path and len(file_path) > 1:
                alert.files.add(file_path)

        alert.urls.update(self.url_pattern.findall(log_text))

    def extract(self, alert_data: Dict) -> Alert:
        if "agent" in alert_data or "data" in alert_data or "rule" in alert_data:
            return self.extract_wazuh(alert_data)
        elif "AnalysisComponent" in alert_data:
            return self.extract_aminer(alert_data)
        else:
            raise ValueError(f"Unknown alert format: {list(alert_data.keys())}")


class AttackPhaseLabeler:
    def __init__(self, attack_windows: List[Tuple[datetime, datetime, str]]):
        self.attack_windows = attack_windows
        self.pre_attack_buffer = timedelta(seconds=30)
        self.post_attack_buffer = timedelta(seconds=60)

    def label_by_time_and_phase(
        self, alert: Alert
    ) -> Tuple[str, float, str, Optional[str]]:
        timestamp = alert.timestamp

        # Two-pass: exhaust direct-hit check before any buffer check. Attack
        # phases here are back-to-back (gaps of 0-30s), so a 60s post-buffer
        # on window N would swallow window N+1 if we short-circuit.
        for start, end, attack_type in self.attack_windows:
            if start <= timestamp <= end:
                return self._classify_in_window(alert, attack_type)

        for start, end, attack_type in self.attack_windows:
            if start - self.pre_attack_buffer <= timestamp < start:
                return ("PRE_ATTACK", 0.8, f"temporal_pre_{attack_type}", attack_type)
            if end < timestamp <= end + self.post_attack_buffer:
                return ("POST_ATTACK", 0.8, f"temporal_post_{attack_type}", attack_type)

        return ("BENIGN", 0.9, "temporal_baseline", None)

    def _classify_in_window(
        self, alert: Alert, attack_type: str
    ) -> Tuple[str, float, str, Optional[str]]:
        BENIGN_RULE_GROUPS = {
            "authentication_success",
            "dovecot",
            "clamd",
            "freshclam",
            "virus",
            "syslog",
            "pam",
        }
        ATTACK_RULE_GROUPS = {
            "web_attack",
            "sql_injection",
            "exploit_kit",
            "shellshock",
            "rootkit",
            "trojans",
        }

        rule_groups_set = set(alert.rule_groups)

        if rule_groups_set & BENIGN_RULE_GROUPS and not (
            rule_groups_set & ATTACK_RULE_GROUPS
        ):
            return ("BENIGN", 0.9, f"rule_group_benign_in_{attack_type}", attack_type)

        if rule_groups_set & ATTACK_RULE_GROUPS:
            return (
                "MALICIOUS",
                0.9,
                f"rule_group_attack_in_{attack_type}",
                attack_type,
            )

        suricata_category = (alert.category or "").lower()

        # Wazuh aggregate IDS rules leave category empty. Classification is in full_log.
        if not suricata_category and alert.full_log:
            cl_match = re.search(r"\[Classification:\s*([^\]]+)\]", alert.full_log)
            if cl_match:
                suricata_category = cl_match.group(1).strip().lower()

        MALICIOUS_SURICATA_CATEGORIES = {
            "network scan",
            "attempted information leak",
            "a network trojan was detected",
            "trojan activity",
            "attempted administrator privilege gain",
            "attempted user privilege gain",
            "successful administrator privilege gain",
            "web application attack",
            "denial of service",
            "shellcode detect",
            "exploit kit activity",
            "malware command and control activity detected",
        }
        NOISY_SURICATA_CATEGORIES = {
            "not suspicious traffic",
            "unknown traffic",
            "misc activity",
            "protocol command decode",
            "tcp connection refused",
        }
        PHASES_WHERE_PBT_IS_SIGNAL = {
            "network_scans",
            "service_scans",
            "wpscan",
            "dirb",
            "cracking",
            "webshell",
        }
        PHASES_WHERE_TLS_ERRORS_ARE_SIGNAL = {
            "cracking",
            "reverse_shell",
            "webshell",
        }

        if suricata_category and any(
            c in suricata_category for c in MALICIOUS_SURICATA_CATEGORIES
        ):
            return ("MALICIOUS", 0.88, f"suricata_category_{attack_type}", attack_type)

        if suricata_category and "generic protocol command decode" in suricata_category:
            if attack_type in PHASES_WHERE_TLS_ERRORS_ARE_SIGNAL:
                return (
                    "SUSPICIOUS",
                    0.65,
                    f"suricata_tls_error_in_{attack_type}",
                    attack_type,
                )
            else:
                return ("BENIGN", 0.80, f"suricata_noisy_in_{attack_type}", attack_type)

        if suricata_category and "potentially bad traffic" in suricata_category:
            if attack_type in PHASES_WHERE_PBT_IS_SIGNAL:
                return (
                    "SUSPICIOUS",
                    0.65,
                    f"suricata_pbt_in_{attack_type}",
                    attack_type,
                )
            else:
                return (
                    "BENIGN",
                    0.75,
                    f"suricata_pbt_noise_in_{attack_type}",
                    attack_type,
                )

        if suricata_category and any(
            c in suricata_category for c in NOISY_SURICATA_CATEGORIES
        ):
            return ("BENIGN", 0.75, f"suricata_noisy_in_{attack_type}", attack_type)

        if alert.source == "aminer":
            sig = (alert.signature or "").lower()

            AMINER_MALICIOUS = {
                "high entropy in dns domain": ("dnsteal", 0.92),
                "new ip address in dns logs": ("network_scans", 0.80),
                "unusual occurrence frequencies of query records in dns": (
                    "dnsteal",
                    0.85,
                ),
                "unusual occurrence frequencies of dns query ips": (
                    "network_scans",
                    0.80,
                ),
                "new characters in apache access request": ("dirb", 0.78),
                "new characters in apache access referer": ("webshell", 0.75),
                "new status code in apache access log": ("dirb", 0.70),
                "cpu value out of expected range": (None, 0.60),
            }
            AMINER_BENIGN = {
                "new service_start parameter combination",
                "new service_stop parameter combination",
                "new user_acct parameter combination",
                "new user_auth parameter combination",
                "new event type",
                "cpu value deviates from average",
            }

            for pattern, (phase, conf) in AMINER_MALICIOUS.items():
                if pattern in sig:
                    # ClamAV update checks generate high-entropy TXT queries to
                    # clamav.net that AMiner legitimately flags but aren't dnsteal.
                    if "dns" in pattern:
                        benign_dns_domains = (
                            "clamav.net",
                            "cvd.clamav.net",
                            "db.local.clamav.net",
                        )
                        raw_log = (alert.full_log or "").lower()
                        if any(d in raw_log for d in benign_dns_domains):
                            return (
                                "BENIGN",
                                0.85,
                                f"aminer_clamav_dns_fp_{attack_type}",
                                attack_type,
                            )
                    if phase is None or phase == attack_type:
                        return (
                            "MALICIOUS",
                            conf,
                            f"aminer_component_{attack_type}",
                            attack_type,
                        )
                    else:
                        return (
                            "SUSPICIOUS",
                            0.55,
                            f"aminer_component_phase_mismatch_{attack_type}",
                            attack_type,
                        )

            for pattern in AMINER_BENIGN:
                if pattern in sig:
                    return (
                        "BENIGN",
                        0.80,
                        f"aminer_benign_component_{attack_type}",
                        attack_type,
                    )

            QUIET_PHASES = {"dnsteal", "service_stop"}
            if attack_type in QUIET_PHASES:
                return (
                    "SUSPICIOUS",
                    0.60,
                    f"aminer_unknown_in_{attack_type}",
                    attack_type,
                )
            else:
                return (
                    "SUSPICIOUS",
                    0.50,
                    f"aminer_unknown_in_{attack_type}",
                    attack_type,
                )

        signature_lower = (alert.signature or "").lower()
        category_lower = (alert.category or "").lower()
        full_log_lower = (alert.full_log or "").lower()
        combined = f"{signature_lower} {category_lower} {full_log_lower}"

        phase_patterns = EnhancedEntityExtractor.COMPILED_PATTERNS.get(attack_type, [])
        phase_matches = sum(1 for p in phase_patterns if p.search(combined))

        if phase_matches > 0:
            confidence = min(0.90, 0.75 + (phase_matches * 0.08))
            return (
                "MALICIOUS",
                confidence,
                f"temporal_phase_match_{attack_type}",
                attack_type,
            )

        # Do NOT add 'forbidden', 'denied', 'blocked', 'unauthorized', 'suspicious'
        # here - they fire on HTTP 403s, SSH failures, firewall drops, AppArmor
        # events: all normal background traffic.
        GENERIC_ATTACK_KEYWORDS = [
            "sql injection",
            "xss",
            "cross-site script",
            "command injection",
            "path traversal",
            "directory traversal",
            "remote code execution",
            "shellcode",
            "exploit",
            "malicious payload",
            "intrusion detected",
            "attack detected",
        ]
        generic_matches = sum(1 for kw in GENERIC_ATTACK_KEYWORDS if kw in combined)

        if generic_matches >= 2:
            confidence = min(0.82, 0.65 + (generic_matches * 0.05))
            return (
                "MALICIOUS",
                confidence,
                f"temporal_generic_match_{attack_type}",
                attack_type,
            )

        BENIGN_KEYWORDS = [
            "clamav",
            "freshclam",
            "apt-get",
            "yum update",
            "dnf update",
            "package update",
            "software update",
            "package management",
            "authentication success",
            "login success",
            "accepted password",
            "accepted publickey",
            "session opened for user",
            "cron",
            "logrotate",
            "rsyslog",
            "journald",
            "systemd",
            "postfix",
            "sshd: server listening",
            "ossec started",
            "wazuh started",
            "integrity checksum",
            "ntpd",
            "chronyd",
            "dhclient",
        ]
        if any(kw in combined for kw in BENIGN_KEYWORDS):
            return (
                "BENIGN",
                0.85,
                f"temporal_benign_keyword_{attack_type}",
                attack_type,
            )

        if alert.severity >= 10:
            return (
                "SUSPICIOUS",
                0.55,
                f"temporal_high_severity_{attack_type}",
                attack_type,
            )

        # Default BENIGN (not SUSPICIOUS) to avoid mass FPs from background
        # traffic overlapping attack windows. Graph propagation upgrades alerts
        # sharing entities with confirmed malicious ones.
        return ("BENIGN", 0.65, f"temporal_in_{attack_type}_no_signal", attack_type)


class AlertLabeler:
    def __init__(self, attack_windows: List[Tuple[datetime, datetime, str]]):
        self.extractor = EnhancedEntityExtractor()
        self.phase_labeler = AttackPhaseLabeler(attack_windows)

    def label_scenario(
        self, alerts_json: List[Dict], scenario_name: str
    ) -> List[Alert]:
        print(f"{scenario_name}: extracting from {len(alerts_json)} alerts")
        alerts = []
        failed = 0

        for alert_json in alerts_json:
            try:
                alerts.append(self.extractor.extract(alert_json))
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"parse fail: {e}")

        if failed > 5:
            print(f"{failed} total parse failures")

        label_counts = defaultdict(int)
        phase_counts = defaultdict(int)

        for alert in alerts:
            label, confidence, method, attack_phase = (
                self.phase_labeler.label_by_time_and_phase(alert)
            )
            alert.label = label
            alert.confidence = confidence
            alert.label_method = method
            alert.attack_phase = attack_phase
            label_counts[label] += 1
            if attack_phase:
                phase_counts[attack_phase] += 1

        for label, count in sorted(label_counts.items()):
            print(f"{label}: {count:,} ({100*count/len(alerts):.1f}%)")
        for phase, count in sorted(phase_counts.items()):
            print(f"phase {phase}: {count:,}")

        return alerts

    def export_comprehensive(self, alerts: List[Alert], output_path: str):
        output_data = []

        for alert in alerts:
            output_data.append(
                {
                    "raw_alert": alert.raw_data,
                    "entities": {
                        "src_ip": alert.src_ip,
                        "dst_ip": alert.dst_ip,
                        "src_port": alert.src_port,
                        "dst_port": alert.dst_port,
                        "domains": sorted(alert.domains),
                        "hostnames": sorted(alert.hostnames),
                        "users": sorted(alert.users),
                        "processes": sorted(alert.processes),
                        "files": sorted(alert.files),
                        "urls": sorted(alert.urls),
                    },
                    "metadata": {
                        "signature": alert.signature,
                        "category": alert.category,
                        "severity": alert.severity,
                        "rule_id": alert.rule_id,
                        "rule_groups": alert.rule_groups,
                        "compliance_tags": alert.compliance_tags,
                        "full_log": alert.full_log,
                    },
                    "ground_truth": {
                        "label": alert.label,
                        "confidence": alert.confidence,
                        "label_method": alert.label_method,
                        "attack_phase": alert.attack_phase,
                        "needs_review": alert.label == "SUSPICIOUS"
                        or alert.confidence < 0.6,
                    },
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source,
                }
            )

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"wrote {len(output_data):,} alerts to {output_path}")


def parse_attack_windows(
    labels_csv_path: str, scenario_name: str
) -> List[Tuple[datetime, datetime, str]]:
    windows = []
    with open(labels_csv_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[0] == scenario_name:
            attack_type = parts[1]
            start = datetime.fromtimestamp(float(parts[2]), tz=timezone.utc)
            end = datetime.fromtimestamp(float(parts[3]), tz=timezone.utc)
            windows.append((start, end, attack_type))

    return windows


if __name__ == "__main__":
    scenario = "fox"

    attack_windows = parse_attack_windows("set/labels.csv", scenario)
    print(f"{len(attack_windows)} attack windows for {scenario}")
    for start, end, attack_type in attack_windows:
        print(f"{attack_type}: {start} -> {end}")

    labeler = AlertLabeler(attack_windows)

    with open(f"set/{scenario}_wazuh.json", "r") as f:
        wazuh_alerts = [json.loads(line) for line in f]
    with open(f"set/{scenario}_aminer.json", "r") as f:
        aminer_alerts = [json.loads(line) for line in f]

    all_alerts = wazuh_alerts + aminer_alerts
    print(f"loaded {len(all_alerts):,} alerts")

    labeled_alerts = labeler.label_scenario(all_alerts, scenario)
    labeler.export_comprehensive(
        labeled_alerts, f"labeled_v2/{scenario}_comprehensive.json"
    )

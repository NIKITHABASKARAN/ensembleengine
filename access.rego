package authz.access
import future.keywords.if

default action := "ALLOW"

action := "BLOCK" if { input.impossible_travel == true }
action := "BLOCK" if { input.risk_score >= 0.85 }
action := "MFA"   if { input.risk_score >= 0.5; input.risk_score < 0.85 }
action := "MFA"   if { input.entropy >= 0.7; input.risk_score >= 0.3 }
action := "MFA"   if { input.device_trusted == false; input.risk_score >= 0.4 }

reason := "impossible_travel" if { action == "BLOCK"; input.impossible_travel }
reason := "high_risk_score"   if { action == "BLOCK"; input.risk_score >= 0.85 }
reason := "elevated_risk"     if { action == "MFA"; input.risk_score >= 0.5 }
reason := "model_uncertainty" if { action == "MFA"; input.entropy >= 0.7 }
reason := "untrusted_device"  if { action == "MFA"; input.device_trusted == false }
reason := "low_risk"          if { action == "ALLOW" }

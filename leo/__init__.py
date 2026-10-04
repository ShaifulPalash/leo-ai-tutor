"""Leo - a multi-agent AI tutor built with CrewAI."""

import os

# These MUST be set before CrewAI is imported anywhere.
# setdefault() means: only set if the user has not already set them.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

__version__ = "1.0.0"

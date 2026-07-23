"""Deprecated alias module. Prefer ``pipeline_whereedit``."""

from pipeline_whereedit import WhereEditPipeline, WhereEditPipelineOutput

ChordEditPipeline = WhereEditPipeline
ChordEditPipelineOutput = WhereEditPipelineOutput

__all__ = ["ChordEditPipeline", "ChordEditPipelineOutput", "WhereEditPipeline", "WhereEditPipelineOutput"]

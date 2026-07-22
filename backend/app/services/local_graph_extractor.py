"""
Local graph extractor — LLM-based entity/relation extraction from text chunks.
Replaces Zep Cloud server-side NLP extraction pipeline (now local LLM-based).
"""

import json
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.embeddings import embed
from .local_graph_store import LocalGraphStore

logger = get_logger("mirofish.extractor")

EXTRACT_SYSTEM_PROMPT = """You are a knowledge-graph extraction engine. Given a text chunk and an ontology (entity types + relation types), extract entities and relations as valid JSON.

Output ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "EntityName", "type": "EntityType", "summary": "One-sentence description", "attributes": {}}
  ],
  "edges": [
    {"name": "relation_type", "fact": "Natural language fact statement", "source": "SourceEntityName", "target": "TargetEntityName", "attributes": {}}
  ]
}

Rules:
1. Only use entity types and relation types defined in the ontology
2. Entity names should be proper nouns or specific identifiers from the text
3. Each edge fact must be a complete sentence describing the relationship
4. If no entities/edges are found, return {"entities": [], "edges": []}
5. Do not include any text outside the JSON object"""


class LocalGraphExtractor:
    """Extracts entities and relations from text using LLM, stores in LocalGraphStore."""

    def __init__(self, graph_id: str, ontology: Dict[str, Any]):
        self.store = LocalGraphStore(graph_id)
        self.ontology = ontology
        self.llm = LLMClient()
        self._entity_index: Dict[str, str] = {}  # name -> uuid

    def extract_and_store(self, chunks: List[str], progress_callback=None) -> int:
        """Extract entities/edges from text chunks, store in graph. Returns total extracted count."""
        entity_types = [e["name"] for e in self.ontology.get("entity_types", [])]
        edge_types = [e["name"] for e in self.ontology.get("edge_types", [])]
        ontology_summary = json.dumps(
            {"entity_types": entity_types, "edge_types": edge_types}, ensure_ascii=False
        )

        total = 0
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            result = self._extract_from_chunk(chunk, ontology_summary)
            if not result:
                continue

            for entity in result.get("entities", []):
                self._add_or_merge_entity(entity)

            for edge in result.get("edges", []):
                self._add_edge(edge)

            total += len(result.get("entities", [])) + len(result.get("edges", []))

        return total

    def _extract_from_chunk(
        self, chunk: str, ontology_summary: str
    ) -> Optional[Dict[str, Any]]:
        user_msg = f"Ontology types: {ontology_summary}\n\nText chunk:\n{chunk}"
        try:
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            return self.llm.chat_json(messages, temperature=0.2, max_tokens=8192)
        except Exception as e:
            logger.warning(f"Extraction failed for chunk: {str(e)[:100]}")
            return None

    def _add_or_merge_entity(self, entity: Dict[str, Any]):
        name = entity.get("name", "").strip()
        etype = entity.get("type", "Entity")
        if not name:
            return

        existing = self.store.find_node_by_name(name, labels=[etype])
        if existing:
            return

        labels = ["Entity", etype] if etype != "Entity" else ["Entity"]
        summary = entity.get("summary", "")
        attributes = entity.get("attributes", {})

        emb = embed(f"{name} {summary}") if summary else embed(name)
        node_uuid = self.store.add_node(
            name=name,
            labels=labels,
            summary=summary,
            attributes=attributes,
            embedding=emb,
        )
        self._entity_index[f"{name}:{etype}"] = node_uuid

    def _add_edge(self, edge: Dict[str, Any]):
        name = edge.get("name", "").strip()
        source_name = edge.get("source", "").strip()
        target_name = edge.get("target", "").strip()
        fact = edge.get("fact", "").strip()

        if not name or not source_name or not target_name or not fact:
            return

        source_uuid = self._find_entity_uuid(source_name)
        target_uuid = self._find_entity_uuid(target_name)

        if not source_uuid or not target_uuid:
            return

        emb = embed(fact)
        self.store.add_edge(
            name=name,
            fact=fact,
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            fact_type=name,
            attributes=edge.get("attributes", {}),
            embedding=emb,
        )

    def _find_entity_uuid(self, name: str) -> Optional[str]:
        for key, uuid in self._entity_index.items():
            if key.startswith(f"{name}:"):
                return uuid
        node = self.store.find_node_by_name(name)
        if node:
            self._entity_index[
                f"{node.name}:{node.labels[-1] if len(node.labels) > 1 else 'Entity'}"
            ] = node.uuid
            return node.uuid
        return None

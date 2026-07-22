"""
Graph retrieval tools service.

Wraps graph search, node reading, edge query and other tools for the
Report Agent to use.

Core retrieval tools (optimized):
1. InsightForge (deep insight retrieval) - the most powerful hybrid retrieval,
   automatically generating sub-questions and retrieving across multiple dimensions.
2. PanoramaSearch (breadth search) - get the full picture, including expired content.
3. QuickSearch (simple search) - fast retrieval.
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t
from ..utils.embeddings import embed, cosine_similarity
from .local_graph_store import LocalGraphStore, LocalNode, LocalEdge

logger = get_logger("mirofish.graph_tools")


@dataclass
class SearchResult:
    """Search result."""

    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        """Convert to text format for the LLM to understand."""
        text_parts = [
            f"Search query: {self.query}",
            f"Found {self.total_count} related pieces of information",
        ]

        if self.facts:
            text_parts.append("\n### Related Facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information."""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    def to_text(self) -> str:
        """Convert to text format."""
        entity_type = next(
            (l for l in self.labels if l not in ["Entity", "Node"]),
            t("api.unknownType"),
        )
        return f"Entity: {self.name} (type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information."""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Time info
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format."""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = (
            f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        )

        if include_temporal:
            valid_at = self.valid_at or t("api.unknown")
            invalid_at = self.invalid_at or "to date"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Whether the edge has expired."""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Whether the edge is invalid."""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep insight retrieval result (InsightForge).

    Contains retrieval results for multiple sub-questions as well as the
    integrated analysis.
    """

    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Retrieval results by dimension
    semantic_facts: List[str] = field(default_factory=list)  # Semantic search results
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entity insights
    relationship_chains: List[str] = field(default_factory=list)  # Relationship chains

    # Statistics
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        """Convert to detailed text format for the LLM to understand."""
        text_parts = [
            f"## Future Prediction Deep Analysis",
            f"Analysis question: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction Data Statistics",
            f"- Related prediction facts: {self.total_facts}",
            f"- Involved entities: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}",
        ]

        # Sub-queries
        if self.sub_queries:
            text_parts.append(f"\n### Sub-questions Analyzed")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # Semantic search results
        if self.semantic_facts:
            text_parts.append(
                f"\n### [Key Facts] (please quote these original texts in the report)"
            )
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Entity insights
        if self.entity_insights:
            text_parts.append(f"\n### [Core Entities]")
            for entity in self.entity_insights:
                text_parts.append(
                    f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})"
                )
                if entity.get("summary"):
                    text_parts.append(f'  Summary: "{entity.get("summary")}"')
                if entity.get("related_facts"):
                    text_parts.append(
                        f"  Related facts: {len(entity.get('related_facts', []))}"
                    )

        # Relationship chains
        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship Chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Breadth search result (Panorama).

    Contains all related information, including expired content.
    """

    query: str

    # All nodes
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # All edges (including expired)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Currently active facts
    active_facts: List[str] = field(default_factory=list)
    # Expired/invalid facts (historical)
    historical_facts: List[str] = field(default_factory=list)

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        """Convert to text format (full version, not truncated)."""
        text_parts = [
            f"## Broad Search Results (Future Panorama View)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Currently valid facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}",
        ]

        # Currently active facts (full output, no truncation)
        if self.active_facts:
            text_parts.append(
                f"\n### [Currently Valid Facts] (original simulation results)"
            )
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Historical/expired facts (full output, no truncation)
        if self.historical_facts:
            text_parts.append(
                f"\n### [Historical/Expired Facts] (evolution process records)"
            )
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Key entities (full output, no truncation)
        if self.all_nodes:
            text_parts.append(f"\n### [Involved Entities]")
            for node in self.all_nodes:
                entity_type = next(
                    (l for l in node.labels if l not in ["Entity", "Node"]), "Entity"
                )
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Interview result for a single Agent."""

    agent_name: str
    agent_role: str  # Role type (e.g. student, teacher, media, etc.)
    agent_bio: str  # Bio
    question: str  # Interview question
    response: str  # Interview response
    key_quotes: List[str] = field(default_factory=list)  # Key quotes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Show full agent_bio, no truncation
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                # Clean various quote marks
                clean_quote = (
                    quote.replace("\u201c", "").replace("\u201d", "").replace('"', "")
                )
                clean_quote = clean_quote.replace("\u300c", "").replace("\u300d", "")
                clean_quote = clean_quote.strip()
                # Strip leading punctuation
                while clean_quote and clean_quote[0] in "，,；;：:、。！？\n\r\t ":
                    clean_quote = clean_quote[1:]
                # Filter garbage content containing question numbers (Q1-9)
                skip = False
                for d in "123456789":
                    if f"\u95ee\u9898{d}" in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncate long content (by sentence, not hard cut)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find("\u3002", 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[: dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview result.

    Contains interview answers from multiple simulated Agents.
    """

    interview_topic: str  # Interview topic
    interview_questions: List[str]  # Interview question list

    # Agents selected for interview
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Interview responses per Agent
    interviews: List[AgentInterview] = field(default_factory=list)

    # Reason for selecting Agents
    selection_reasoning: str = ""
    # Consolidated interview summary
    summary: str = ""

    # Statistics
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        """Convert to detailed text format for the LLM to understand and quote in the report."""
        text_parts = [
            "## Deep Interview Report",
            f"**Interview topic:** {self.interview_topic}",
            f"**Interview count:** {self.interviewed_count} / {self.total_agents} simulation agents",
            "\n### Interview Subject Selection Rationale",
            self.selection_reasoning or "(auto-selected)",
            "\n---",
            "\n### Interview Transcript",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview Summary and Core Viewpoints")
        text_parts.append(self.summary or "(No summary)")

        return "\n".join(text_parts)


class GraphToolsService:
    """
    Graph retrieval tools service.

    [Core retrieval tools - optimized]
    1. insight_forge - deep insight retrieval (most powerful; auto-generates
       sub-questions and retrieves across multiple dimensions).
    2. panorama_search - breadth search (gets the full picture, including
       expired content).
    3. quick_search - simple search (fast retrieval).
    4. interview_agents - deep interview (interviews simulated Agents to obtain
       multi-perspective views).

    [Basic tools]
    - search_graph - semantic graph search.
    - get_all_nodes - get all nodes in the graph.
    - get_all_edges - get all edges in the graph (including temporal info).
    - get_node_detail - get detailed information of a node.
    - get_node_edges - get edges related to a node.
    - get_entities_by_type - get entities by type.
    - get_entity_summary - get a relationship summary for an entity.
    """

    def __init__(
        self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None
    ):
        # LLM client for InsightForge sub-query generation
        self._llm_client = llm_client
        logger.info(t("console.graphToolsInitialized"))

    @property
    def llm(self) -> LLMClient:
        """Lazily initialize the LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def search_graph(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        Semantic graph search.

        Uses the local graph store for searching:
        - If a node/edge has an embedding, re-rank embed(query) with cosine_similarity.
        - Otherwise use SQL LIKE keyword matching (already implemented by the store).

        Args:
            graph_id: Graph ID (Standalone Graph).
            query: Search query.
            limit: Number of results to return.
            scope: Search scope, "edges" or "nodes".

        Returns:
            SearchResult: The search result.
        """
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))

        store = LocalGraphStore(graph_id)
        facts = []
        edges = []
        nodes = []

        # Try to get query embedding for semantic reranking
        query_embedding = embed(query)

        if scope in ("edges", "both"):
            local_edges = store.search_edges(query, limit)
            # If embedding available, rerank with cosine_similarity
            if query_embedding is not None:
                scored = []
                for e in local_edges:
                    if e.embedding:
                        sim = cosine_similarity(query_embedding, e.embedding)
                    else:
                        sim = 0.0
                    scored.append((sim, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                local_edges = [e for _, e in scored[:limit]]

            for e in local_edges:
                if e.fact:
                    facts.append(e.fact)
                edges.append(
                    {
                        "uuid": e.uuid,
                        "name": e.name,
                        "fact": e.fact,
                        "source_node_uuid": e.source_node_uuid,
                        "target_node_uuid": e.target_node_uuid,
                    }
                )

        if scope in ("nodes", "both"):
            local_nodes = store.search_nodes(query, limit)
            if query_embedding is not None:
                scored = []
                for n in local_nodes:
                    if n.embedding:
                        sim = cosine_similarity(query_embedding, n.embedding)
                    else:
                        sim = 0.0
                    scored.append((sim, n))
                scored.sort(key=lambda x: x[0], reverse=True)
                local_nodes = [n for _, n in scored[:limit]]

            for n in local_nodes:
                nodes.append(
                    {
                        "uuid": n.uuid,
                        "name": n.name,
                        "labels": n.labels,
                        "summary": n.summary,
                    }
                )
                if n.summary:
                    facts.append(f"[{n.name}]: {n.summary}")

        logger.info(t("console.searchComplete", count=len(facts)))

        return SearchResult(
            facts=facts,
            edges=edges,
            nodes=nodes,
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes in the graph.

        Args:
            graph_id: Graph ID.

        Returns:
            List of nodes.
        """
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        store = LocalGraphStore(graph_id)
        nodes = store.get_all_nodes()

        result = []
        for node in nodes:
            result.append(
                NodeInfo(
                    uuid=node.uuid or "",
                    name=node.name or "",
                    labels=node.labels or [],
                    summary=node.summary or "",
                    attributes=node.attributes or {},
                )
            )

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(
        self, graph_id: str, include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """
        Get all edges in the graph (including temporal information).

        Args:
            graph_id: Graph ID.
            include_temporal: Whether to include temporal info (default True).

        Returns:
            List of edges (including created_at, valid_at, invalid_at, expired_at).
        """
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        store = LocalGraphStore(graph_id)
        edges = store.get_all_edges()

        result = []
        for edge in edges:
            edge_info = EdgeInfo(
                uuid=edge.uuid or "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or "",
            )

            if include_temporal:
                edge_info.created_at = edge.created_at
                edge_info.valid_at = edge.valid_at
                edge_info.invalid_at = edge.invalid_at
                edge_info.expired_at = edge.expired_at

            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result

    def get_node_detail(self, graph_id: str, node_uuid: str) -> Optional[NodeInfo]:
        """
        Get detailed information of a single node.

        Args:
            graph_id: Graph ID.
            node_uuid: Node UUID.

        Returns:
            Node information, or None.
        """
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))

        try:
            store = LocalGraphStore(graph_id)
            node = store.get_node(node_uuid)

            if not node:
                return None

            return NodeInfo(
                uuid=node.uuid or "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
            )
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to a node.

        Args:
            graph_id: Graph ID.
            node_uuid: Node UUID.

        Returns:
            List of edges.
        """
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))

        try:
            store = LocalGraphStore(graph_id)
            edges = store.get_node_edges(node_uuid)

            result = []
            for edge in edges:
                edge_info = EdgeInfo(
                    uuid=edge.uuid or "",
                    name=edge.name or "",
                    fact=edge.fact or "",
                    source_node_uuid=edge.source_node_uuid or "",
                    target_node_uuid=edge.target_node_uuid or "",
                    created_at=edge.created_at,
                    valid_at=edge.valid_at,
                    invalid_at=edge.invalid_at,
                    expired_at=edge.expired_at,
                )
                result.append(edge_info)

            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result

        except Exception as e:
            logger.warning(t("console.fetchNodeEdgesFailed", error=str(e)))
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """
        Get entities by type.

        Args:
            graph_id: Graph ID.
            entity_type: Entity type (e.g. Student, PublicFigure, etc.).

        Returns:
            List of entities matching the type.
        """
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            # Check if labels contain specified type
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(
            t("console.foundEntitiesByType", count=len(filtered), type=entity_type)
        )
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """
        Get the relationship summary for a given entity.

        Searches all information related to that entity and generates a summary.

        Args:
            graph_id: Graph ID.
            entity_name: Entity name.

        Returns:
            Entity summary information.
        """
        logger.info(t("console.fetchingEntitySummary", name=entity_name))

        # First search for info related to this entity
        search_result = self.search_graph(
            graph_id=graph_id, query=entity_name, limit=20
        )

        # Try to find this entity among all nodes
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            # Pass graph_id parameter
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get statistics for the graph.

        Args:
            graph_id: Graph ID.

        Returns:
            Statistics information.
        """
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        # Entity type distribution statistics
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        # Relation type distribution statistics
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_simulation_context(
        self, graph_id: str, simulation_requirement: str, limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get simulation-related context information.

        Comprehensively searches all information related to the simulation
        requirement.

        Args:
            graph_id: Graph ID.
            simulation_requirement: Description of the simulation requirement.
            limit: Per-category information count limit.

        Returns:
            Simulation context information.
        """
        logger.info(
            t("console.fetchingSimContext", requirement=simulation_requirement[:50])
        )

        # Search for info related to simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id, query=simulation_requirement, limit=limit
        )

        # Get graph statistics
        stats = self.get_graph_statistics(graph_id)

        # Get all entity nodes
        all_nodes = self.get_all_nodes(graph_id)

        # Filter entities with actual types (not bare Entity nodes)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append(
                    {
                        "name": node.name,
                        "type": custom_labels[0],
                        "summary": node.summary,
                    }
                )

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limit count
            "total_entities": len(entities),
        }

    # ========== Core Retrieval Tools (Optimized) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        [InsightForge - deep insight retrieval]

        The most powerful hybrid retrieval function; automatically decomposes
        the question and retrieves across multiple dimensions:
        1. Use the LLM to decompose the question into multiple sub-questions.
        2. Perform a semantic search for each sub-question.
        3. Extract related entities and obtain their detailed information.
        4. Trace relationship chains.
        5. Integrate all results to generate a deep insight.

        Args:
            graph_id: Graph ID.
            query: User question.
            simulation_requirement: Description of the simulation requirement.
            report_context: Report context (optional; used for more precise
                sub-question generation).
            max_sub_queries: Maximum number of sub-questions.

        Returns:
            InsightForgeResult: The deep insight retrieval result.
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))

        result = InsightForgeResult(
            query=query, simulation_requirement=simulation_requirement, sub_queries=[]
        )

        # Step 1: Use LLM to generate sub-queries
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))

        # Step 2: Semantic search for each sub-query
        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id, query=sub_query, limit=15, scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        # Also search the original query
        main_search = self.search_graph(
            graph_id=graph_id, query=query, limit=20, scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: Extract related entity UUIDs from edges, only fetch these entities (not all nodes)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # Get details of all related entities (no limit, full output)
        entity_insights = []
        node_map = {}  # For later relationship chain construction

        for uuid in list(entity_uuids):  # Process all entities, no truncation
            if not uuid:
                continue
            try:
                # Get info for each related node individually
                node = self.get_node_detail(graph_id, uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next(
                        (l for l in node.labels if l not in ["Entity", "Node"]),
                        "Entity",
                    )

                    # Get all facts related to this entity (no truncation)
                    related_facts = [
                        f for f in all_facts if node.name.lower() in f.lower()
                    ]

                    entity_insights.append(
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "type": entity_type,
                            "summary": node.summary,
                            "related_facts": related_facts,  # Full output, no truncation
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to get node {uuid}: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: Build all relationship chains (no limit)
        relationship_chains = []
        for edge_data in all_edges:  # Process all edges, no truncation
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                relation_name = edge_data.get("name", "")

                source_name = (
                    node_map.get(source_uuid, NodeInfo("", "", [], "", {})).name
                    or source_uuid[:8]
                )
                target_name = (
                    node_map.get(target_uuid, NodeInfo("", "", [], "", {})).name
                    or target_uuid[:8]
                )

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(
            t(
                "console.insightForgeComplete",
                facts=result.total_facts,
                entities=result.total_entities,
                relationships=result.total_relationships,
            )
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """
        Use the LLM to generate sub-questions.

        Decomposes a complex question into multiple sub-questions that can be
        retrieved independently.
        """
        system_prompt = """You are a professional question analysis expert. Your task is to decompose a complex question into multiple sub-questions that can be independently observed in the simulation world.

Requirements:
1. Each sub-question should be specific enough to find related Agent behaviors or events in the simulation world
2. Sub-questions should cover different dimensions of the original question (e.g.: who, what, why, how, when, where)
3. Sub-questions should be relevant to the simulation scenario
4. Return JSON format: {"sub_queries": ["sub-question1", "sub-question2", ...]}"""

        user_prompt = f"""Simulation requirement background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following question into {max_queries} sub-questions:
{query}

Return the list of sub-questions in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            sub_queries = response.get("sub_queries", [])
            # Ensure it's a list of strings
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            # Fallback: return variants based on original query
            return [
                query,
                f"Main participants of {query}",
                f"Causes and impacts of {query}",
                f"Development process of {query}",
            ][:max_queries]

    def panorama_search(
        self, graph_id: str, query: str, include_expired: bool = True, limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - breadth search]

        Gets the full-picture view, including all related content and
        historical/expired information:
        1. Get all related nodes.
        2. Get all edges (including expired/invalid ones).
        3. Categorize and organize current and historical information.

        This tool is suitable for scenarios where you need to understand the
        full picture of an event or trace its evolution.

        Args:
            graph_id: Graph ID.
            query: Search query (used for relevance ranking).
            include_expired: Whether to include expired content (default True).
            limit: Return result count limit.

        Returns:
            PanoramaResult: The breadth search result.
        """
        logger.info(t("console.panoramaSearchStart", query=query[:50]))

        result = PanoramaResult(query=query)

        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Get all edges (with time info)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Classify facts
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # Add entity names to facts
            source_name = (
                node_map.get(edge.source_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.source_node_uuid[:8]
            )
            target_name = (
                node_map.get(edge.target_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.target_node_uuid[:8]
            )

            # Determine if expired/invalid
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                # Historical/expired facts, add time marker
                valid_at = edge.valid_at or t("api.unknown")
                invalid_at = edge.invalid_at or edge.expired_at or t("api.unknown")
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Currently active facts
                active_facts.append(edge.fact)

        # Relevance sort based on query
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
            if len(w.strip()) > 1
        ]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        # Sort and limit count
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            t(
                "console.panoramaSearchComplete",
                active=result.active_count,
                historical=result.historical_count,
            )
        )
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """
        [QuickSearch - simple search]

        A fast, lightweight retrieval tool:
        1. Directly calls graph semantic search.
        2. Returns the most relevant results.
        3. Suitable for simple, direct retrieval needs.

        Args:
            graph_id: Graph ID.
            query: Search query.
            limit: Number of results to return.

        Returns:
            SearchResult: The search result.
        """
        logger.info(t("console.quickSearchStart", query=query[:50]))

        # Directly call existing search_graph method
        result = self.search_graph(
            graph_id=graph_id, query=query, limit=limit, scope="edges"
        )

        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """
        [InterviewAgents - deep interview]

        Calls the real OASIS interview API to interview Agents that are
        currently running in the simulation:
        1. Automatically read persona files to understand all simulated Agents.
        2. Use the LLM to analyze the interview requirement and intelligently
           select the most relevant Agents.
        3. Use the LLM to generate interview questions.
        4. Call the /api/simulation/interview/batch interface to conduct real
           interviews (interviewing both platforms simultaneously).
        5. Integrate all interview results to generate an interview report.

        [Important] This feature requires the simulation environment to be
        running (the OASIS environment must not be closed).

        [Use cases]
        - Need to understand views on an event from different role perspectives.
        - Need to collect multiple opinions and standpoints.
        - Need to obtain real answers from simulated Agents (not LLM simulation).

        Args:
            simulation_id: Simulation ID (used to locate persona files and call
                the interview API).
            interview_requirement: Interview requirement description (unstructured,
                e.g. "Understand students' views on the incident").
            simulation_requirement: Simulation requirement background (optional).
            max_agents: Maximum number of Agents to interview.
            custom_questions: Custom interview questions (optional; auto-generated
                if not provided).

        Returns:
            InterviewResult: The interview result.
        """
        from .simulation_runner import SimulationRunner

        logger.info(
            t("console.interviewAgentsStart", requirement=interview_requirement[:50])
        )

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: Read persona files
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "No agent persona files available for interview found"
            return result

        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))

        # Step 2: Use LLM to select Agents to interview (returns agent_id list)
        selected_agents, selected_indices, selection_reasoning = (
            self._select_agents_for_interview(
                profiles=profiles,
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                max_agents=max_agents,
            )
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(
            t(
                "console.selectedAgentsForInterview",
                count=len(selected_agents),
                indices=selected_indices,
            )
        )

        # Step 3: Generate interview questions (if not provided)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info(
                t(
                    "console.generatedInterviewQuestions",
                    count=len(result.interview_questions),
                )
            )

        # Merge questions into one interview prompt
        combined_prompt = "\n".join(
            [f"{i + 1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        # Add optimization prefix to constrain Agent reply format
        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Please draw on your persona, all past memories and actions, "
            "and answer the following questions directly in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language, do not call any tools\n"
            "2. Do not return JSON format or tool-call format\n"
            "3. Do not use Markdown headings (e.g. #, ##, ###)\n"
            "4. Answer each question in turn, starting each answer with 'Question X:' (X is the question number)\n"
            "5. Separate answers to each question with a blank line\n"
            "6. Answers should be substantive, with at least 2-3 sentences per question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: Call real interview API (no platform specified, default dual-platform interview)
        try:
            # Build batch interview list (no platform specified, dual-platform interview)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append(
                    {
                        "agent_id": agent_idx,
                        "prompt": optimized_prompt,  # Use optimized prompt
                        # No platform specified, API will interview on both twitter and reddit
                    }
                )

            logger.info(
                t("console.callingBatchInterviewApi", count=len(interviews_request))
            )

            # Call SimulationRunner batch interview method (no platform, dual-platform interview)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # No platform specified, dual-platform interview
                timeout=180.0,  # Dual platform needs longer timeout
            )

            logger.info(
                t(
                    "console.interviewApiReturned",
                    count=api_result.get("interviews_count", 0),
                    success=api_result.get("success"),
                )
            )

            # Check if API call succeeded
            if not api_result.get("success", False):
                error_msg = api_result.get("error", t("api.unknownError"))
                logger.warning(
                    t("console.interviewApiReturnedFailure", error=error_msg)
                )
                result.summary = f"Interview API call failed: {error_msg}. Please check the OASIS simulation environment status."
                return result

            # Step 5: Parse API results, build AgentInterview objects
            # Dual-platform return format: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = (
                api_data.get("results", {}) if isinstance(api_data, dict) else {}
            )

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get(
                    "realname", agent.get("username", f"Agent_{agent_idx}")
                )
                agent_role = agent.get("profession", t("api.unknown"))
                agent_bio = agent.get("bio", "")

                # Get this Agent's interview results from both platforms
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Clean possible tool call JSON wrapper
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Always output dual-platform markers
                twitter_text = (
                    twitter_response
                    if twitter_response
                    else "(No response was obtained from this platform)"
                )
                reddit_text = (
                    reddit_response
                    if reddit_response
                    else "(No response was obtained from this platform)"
                )
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"

                # Extract key quotes (from both platform responses)
                import re

                combined_responses = f"{twitter_response} {reddit_response}"

                # Clean response text: remove markers, numbering, Markdown noise
                clean_text = re.sub(r"#{1,6}\s+", "", combined_responses)
                clean_text = re.sub(r"\{[^}]*tool_name[^}]*\}", "", clean_text)
                clean_text = re.sub(r"[*_`|>~\-]{2,}", "", clean_text)
                clean_text = re.sub(
                    r"(Question|Question)\s*\d+\s*[：:]\s*",
                    "",
                    clean_text,
                    flags=re.IGNORECASE,
                )
                clean_text = re.sub(r"【[^】]+】", "", clean_text)
                clean_text = re.sub(r"\[[^\]]+\]", "", clean_text)

                # Strategy 1 (primary): extract complete substantive sentences
                sentences = re.split(r"[。！？]", clean_text)
                meaningful = [
                    s.strip()
                    for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r"^[\s\W，,；;：:、]+", s.strip())
                    and not s.strip().startswith(("{", "Question", "Question"))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # Strategy 2 (supplement): correctly paired Chinese quote marks「」long text
                if not key_quotes:
                    paired = re.findall(
                        r"\u201c([^\u201c\u201d]{15,100})\u201d", clean_text
                    )
                    paired += re.findall(
                        r"\u300c([^\u300c\u300d]{15,100})\u300d", clean_text
                    )
                    key_quotes = [
                        q for q in paired if not re.match(r"^[，,；;：:、]", q)
                    ][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Expand bio length limit
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            # Simulation environment not running
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"Interview failed: {str(e)}. The simulation environment may be closed; please ensure the OASIS environment is running."
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback

            logger.error(traceback.format_exc())
            result.summary = f"An error occurred during the interview: {str(e)}"
            return result

        # Step 6: Generate interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(
            t("console.interviewAgentsComplete", count=result.interviewed_count)
        )
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Clean the JSON tool-call wrapper in an Agent reply and extract the actual content."""
        if not response or not response.strip().startswith("{"):
            return response
        text = response.strip()
        if "tool_name" not in text[:80]:
            return response
        import re as _re

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "arguments" in data:
                for key in ("content", "text", "body", "message", "reply"):
                    if key in data["arguments"]:
                        return str(data["arguments"][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace("\\n", "\n").replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load the simulated Agent persona files."""
        import os
        import csv

        # Build persona file paths
        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        profiles = []

        # Try reading Reddit JSON format first
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))

        # Try reading Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV format to unified format
                        profiles.append(
                            {
                                "realname": row.get("name", ""),
                                "username": row.get("username", ""),
                                "bio": row.get("description", ""),
                                "persona": row.get("user_char", ""),
                                "profession": t("api.unknown"),
                            }
                        )
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """
        Use the LLM to select the Agents to interview.

        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: list of full info for selected Agents.
                - selected_indices: list of indices for selected Agents (used for API calls).
                - reasoning: selection reasoning.
        """

        # Build Agent summary list
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", t("api.unknown")),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            }
            agent_summaries.append(summary)

        system_prompt = """You are a professional interview planning expert. Your task is to select the most suitable interview subjects from the list of simulation Agents based on the interview requirement.

Selection criteria:
1. The Agent's identity/profession is relevant to the interview topic
2. The Agent may hold unique or valuable viewpoints
3. Select diverse perspectives (e.g.: supporters, opponents, neutrals, professionals, etc.)
4. Prioritize roles directly related to the event

Return JSON format:
{
    "selected_indices": [list of selected Agent indices],
    "reasoning": "explanation of selection reasoning"
}"""

        user_prompt = f"""Interview requirement:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

List of selectable Agents (total {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} most suitable Agents to interview, and explain the selection reasoning."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Auto-selected based on relevance")

            # Get full info for selected Agents
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(t("console.llmSelectAgentFailed", error=e))
            # Fallback: select first N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Use the LLM to generate interview questions."""

        agent_roles = [a.get("profession", t("api.unknown")) for a in selected_agents]

        system_prompt = """You are a professional journalist/interviewer. Based on the interview requirement, generate 3-5 in-depth interview questions.

Question requirements:
1. Open-ended questions that encourage detailed answers
2. May yield different answers for different roles
3. Cover multiple dimensions such as facts, opinions, feelings, etc.
4. Natural language, like a real interview
5. Keep each question within 50 words, concise and clear
6. Ask directly, do not include background explanation or prefixes

Return JSON format: {"questions": ["question1", "question2", ...]}"""

        user_prompt = f"""Interview requirement: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee roles: {", ".join(agent_roles)}

Please generate 3-5 interview questions."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )

            return response.get(
                "questions", [f"What are your views on {interview_requirement}?"]
            )

        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                f"What is your opinion on {interview_requirement}?",
                "What impact does this matter have on you or the group you represent?",
                "How do you think this problem should be solved or improved?",
            ]

    def _generate_interview_summary(
        self, interviews: List[AgentInterview], interview_requirement: str
    ) -> str:
        """Generate an interview summary."""

        if not interviews:
            return "No interviews were completed"

        # Collect all interview content
        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}"
            )

        quote_instruction = (
            "Use Chinese quote marks「」when quoting interviewees"
            if get_locale() == "zh"  # kept for backward compat with Chinese quote marks in interview content
            else "Use quotation marks to quote interviewees."
        )
        system_prompt = f"""You are a professional news editor. Based on the answers of multiple interviewees, generate an interview summary.

Summary requirements:
1. Distill the main viewpoints of each party
2. Point out consensus and disagreements among viewpoints
3. Highlight valuable quotes
4. Objective and neutral, not favoring any party
5. Keep within 1000 words

Format constraints (must be followed):
- Use plain text paragraphs, separate different parts with blank lines
- Do not use Markdown headings (e.g. #, ##, ###)
- Do not use horizontal rules (e.g. ---, ***)
- {quote_instruction}
- You may use **bold** to mark key words, but do not use other Markdown syntax"""

        user_prompt = f"""Interview topic: {interview_requirement}

Interview content:
{"".join(interview_texts)}

Please generate an interview summary."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return summary

        except Exception as e:
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            # Fallback: simple concatenation
            return (
                f"A total of {len(interviews)} interviewees were interviewed, including: "
                + ", ".join([i.agent_name for i in interviews])
            )

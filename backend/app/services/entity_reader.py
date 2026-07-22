"""
Entity reader — reads and filters entities from local sqlite graph store.
Replaces the former Zep-based entity reader (now uses local SQLite).
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from .local_graph_store import LocalGraphStore

logger = get_logger("mirofish.entity_reader")


@dataclass
class EntityNode:
    """Entity node data structure"""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Get entity type (excluding default Entity label)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity set"""

    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    Entity reader and filter service

    Reads nodes from the local graph store, filters by defined entity types.
    Kept class name for import compatibility.
    """

    def __init__(self, api_key: Optional[str] = None):
        pass

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all nodes from graph"""
        logger.info(f"Reading all nodes from graph {graph_id}...")
        store = LocalGraphStore(graph_id)
        nodes = store.get_all_nodes()
        nodes_data = [n.to_dict() for n in nodes]
        logger.info(f"Got {len(nodes_data)} nodes")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all edges from graph"""
        logger.info(f"Reading all edges from graph {graph_id}...")
        store = LocalGraphStore(graph_id)
        edges = store.get_all_edges()
        edges_data = []
        for edge in edges:
            edges_data.append(
                {
                    "uuid": edge.uuid,
                    "name": edge.name,
                    "fact": edge.fact,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes,
                }
            )
        logger.info(f"Got {len(edges_data)} edges")
        return edges_data

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        """Get edges for a specific node"""
        store = LocalGraphStore(graph_id)
        edges = store.get_node_edges(node_uuid)
        edges_data = []
        for edge in edges:
            edges_data.append(
                {
                    "uuid": edge.uuid,
                    "name": edge.name,
                    "fact": edge.fact,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes,
                }
            )
        return edges_data

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """Filter nodes matching defined entity types"""
        logger.info(f"Filtering entities in graph {graph_id}...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [
                    l for l in custom_labels if l in defined_entity_types
                ]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append(
                            {
                                "direction": "outgoing",
                                "edge_name": edge["name"],
                                "fact": edge["fact"],
                                "target_node_uuid": edge["target_node_uuid"],
                            }
                        )
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append(
                            {
                                "direction": "incoming",
                                "edge_name": edge["name"],
                                "fact": edge["fact"],
                                "source_node_uuid": edge["source_node_uuid"],
                            }
                        )
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        rn = node_map[related_uuid]
                        related_nodes.append(
                            {
                                "uuid": rn["uuid"],
                                "name": rn["name"],
                                "labels": rn["labels"],
                                "summary": rn.get("summary", ""),
                            }
                        )
                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(
            f"Filter complete: {total_count} total, {len(filtered_entities)} matched, types: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self, graph_id: str, entity_uuid: str
    ) -> Optional[EntityNode]:
        """Get single entity with full context (edges + related nodes)"""
        store = LocalGraphStore(graph_id)
        node = store.get_node(entity_uuid)
        if not node:
            return None

        edges = self.get_node_edges(graph_id, entity_uuid)
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n["uuid"]: n for n in all_nodes}

        related_edges = []
        related_node_uuids = set()

        for edge in edges:
            if edge["source_node_uuid"] == entity_uuid:
                related_edges.append(
                    {
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    }
                )
                related_node_uuids.add(edge["target_node_uuid"])
            else:
                related_edges.append(
                    {
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    }
                )
                related_node_uuids.add(edge["source_node_uuid"])

        related_nodes = []
        for related_uuid in related_node_uuids:
            if related_uuid in node_map:
                rn = node_map[related_uuid]
                related_nodes.append(
                    {
                        "uuid": rn["uuid"],
                        "name": rn["name"],
                        "labels": rn["labels"],
                        "summary": rn.get("summary", ""),
                    }
                )

        return EntityNode(
            uuid=node.uuid,
            name=node.name,
            labels=node.labels,
            summary=node.summary,
            attributes=node.attributes,
            related_edges=related_edges,
            related_nodes=related_nodes,
        )

    def get_entities_by_type(
        self, graph_id: str, entity_type: str, enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """Get all entities of a specific type"""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities

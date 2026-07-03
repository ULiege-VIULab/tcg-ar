import sys
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pygame
import numpy as np
import pyrr
from pygltflib import GLTF2
import struct
from PIL import Image
import copy

background_vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;

        out vec2 v_texture;

        void main()
        {
            gl_Position = vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """

background_fragment_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

def compile_shader(texture_to_load, normal_to_load, joint_to_load, nb_usefull_joints):
    nb_usefull_joints_str = str(nb_usefull_joints)

    if texture_to_load and normal_to_load and joint_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        layout(location = 2) in vec3 a_normal;
        layout(location = 3) in vec4 a_joint;
        layout(location = 4) in vec4 a_weight;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;
        uniform mat4 u_jointMatrix[%s];

        out vec2 v_texture;

        void main()
        {
            mat4 skinMatrix = 
                a_weight.x * u_jointMatrix[int(a_joint.x)] +
                a_weight.y * u_jointMatrix[int(a_joint.y)] +
                a_weight.z * u_jointMatrix[int(a_joint.z)] +
                a_weight.w * u_jointMatrix[int(a_joint.w)];
            gl_Position = projection * view * zoom * model * skinMatrix * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """ % (nb_usefull_joints_str)

        fragment_src = """
        # version 330

        in vec2 v_texture;

        out vec4 out_color;

        uniform sampler2D s_texture;

        void main()
        {
            out_color = texture(s_texture, v_texture);
        }
        """
    elif texture_to_load and joint_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        layout(location = 2) in vec4 a_joint;
        layout(location = 3) in vec4 a_weight;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;
        uniform mat4 u_jointMatrix[%s];

        out vec2 v_texture;

        void main()
        {
            mat4 skinMatrix = 
                a_weight.x * u_jointMatrix[int(a_joint.x)] +
                a_weight.y * u_jointMatrix[int(a_joint.y)] +
                a_weight.z * u_jointMatrix[int(a_joint.z)] +
                a_weight.w * u_jointMatrix[int(a_joint.w)];
            gl_Position = projection * view * zoom * model * skinMatrix * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """ % (nb_usefull_joints_str)

        fragment_src = """
        # version 330

        in vec2 v_texture;

        out vec4 out_color;

        uniform sampler2D s_texture;

        void main()
        {
            out_color = texture(s_texture, v_texture);
        }
        """
    elif normal_to_load and joint_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec3 a_normal;
        layout(location = 2) in vec4 a_joint;
        layout(location = 3) in vec4 a_weight;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;
        uniform mat4 u_jointMatrix[%s];

        void main()
        {
            mat4 skinMatrix = 
                a_weight.x * u_jointMatrix[int(a_joint.x)] +
                a_weight.y * u_jointMatrix[int(a_joint.y)] +
                a_weight.z * u_jointMatrix[int(a_joint.z)] +
                a_weight.w * u_jointMatrix[int(a_joint.w)];
            gl_Position = projection * view * zoom * model * skinMatrix * vec4(a_position, 1.0);
        }
        """ % (nb_usefull_joints_str)

        fragment_src = """
        # version 330

        out vec4 out_color;

        void main()
        {
            out_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        """
    elif normal_to_load and texture_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        layout(location = 2) in vec3 a_normal;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;

        out vec2 v_texture;

        void main()
        {
            gl_Position = projection * view * zoom * model * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """

        fragment_src = """
        # version 330

        in vec2 v_texture;

        out vec4 out_color;

        uniform sampler2D s_texture;

        void main()
        {
            out_color = texture(s_texture, v_texture);
        }
        """
    elif texture_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;

        out vec2 v_texture;

        void main()
        {
            gl_Position = projection * view * zoom * model * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """

        fragment_src = """
        # version 330

        in vec2 v_texture;

        out vec4 out_color;

        uniform sampler2D s_texture;

        void main()
        {
            out_color = texture(s_texture, v_texture);
        }
        """
    elif normal_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec3 a_normal;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;

        void main()
        {
            gl_Position = projection * view * zoom * model * vec4(a_position, 1.0);
        }
        """

        fragment_src = """
        # version 330

        out vec4 out_color;

        void main()
        {
            out_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        """
    elif joint_to_load:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec4 a_joint;
        layout(location = 2) in vec4 a_weight;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;
        uniform mat4 u_jointMatrix[%s];

        void main()
        {
            mat4 skinMatrix = 
                a_weight.x * u_jointMatrix[int(a_joint.x)] +
                a_weight.y * u_jointMatrix[int(a_joint.y)] +
                a_weight.z * u_jointMatrix[int(a_joint.z)] +
                a_weight.w * u_jointMatrix[int(a_joint.w)];
            gl_Position = projection * view * zoom * model * skinMatrix * vec4(a_position, 1.0);
        }
        """ % (nb_usefull_joints_str)

        fragment_src = """
        # version 330

        out vec4 out_color;

        void main()
        {
            out_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        """
    else:
        vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 zoom;

        void main()
        {
            gl_Position = projection * view * zoom * model * vec4(a_position, 1.0);
        }
        """

        fragment_src = """
        # version 330

        out vec4 out_color;

        void main()
        {
            out_color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        """

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

    return shader

gltf_type = {
        "SCALAR" : 1,
        "VEC2" : 2,
        "VEC3" : 3,
        "VEC4" : 4,
        'MAT2': 4,
        'MAT3': 9,
        "MAT4" : 16
    }

gltf_component_type = {
    "5120" : {
        "nb_byte" : 1,
        "character_converter" : "c"
    },
    "5121" : {
        "nb_byte" : 1,
        "character_converter" : "B"
    },
    "5122" : {
        "nb_byte" : 2,
        "character_converter" : "h"
    },
    "5123" : {
        "nb_byte" : 2,
        "character_converter" : "H"
    },
    "5125" : {
        "nb_byte" : 4,
        "character_converter" : "I"
    },
    "5126" : {
        "nb_byte" : 4,
        "character_converter" : "f"
    },
}

gltf_render_mode = {
    "0" : GL_POINTS,
    "1" : GL_LINES,
    "2" : GL_LINE_LOOP,
    "3" : GL_LINE_STRIP,
    "4" : GL_TRIANGLES,
    "5" : GL_TRIANGLE_STRIP,
    "6" : GL_TRIANGLE_FAN
}

def get_accessor_data(gltf, accesor_number):
    accessor = gltf.accessors[accesor_number]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    nb_item = gltf_type[accessor.type]
    item_nb_byte = gltf_component_type[str(accessor.componentType)]["nb_byte"]
    item_char_converter = gltf_component_type[str(accessor.componentType)]["character_converter"]
    str_converter = "<" + item_char_converter * nb_item

    if bufferView.byteStride:
        item_offset = bufferView.byteStride
    else:
        item_offset = nb_item*item_nb_byte

    nb_byte = nb_item*item_nb_byte

    return accessor, bufferView, data, str_converter, item_offset, nb_byte
    
def load_texture(path, texture, gltf_samplers):
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, gltf_samplers.wrapS)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, gltf_samplers.wrapT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gltf_samplers.minFilter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gltf_samplers.magFilter)

    # load image
    image = Image.open(path)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture

def generate_node_pass(gltf):
    nodes_path = []

    for node_number in range(len(gltf.nodes)):
        node_path = []
        explored_children = [0]
        depth = -1
        nodes_tree = gltf.nodes
        current_node = -1

        node_roots = []

        for scene in gltf.scenes:
            node_roots = node_roots + scene.nodes

        while current_node != node_number:
            #root node reach without finding the target
            if depth == -1:
                next_root = node_roots.pop(0)
                if next_root == node_number:
                    node_path.append(next_root)
                    break
                elif not(nodes_tree[next_root].children):
                    continue
                node_path.append(next_root)
                depth += 1
                current_node = next_root
            #leaf
            if len(nodes_tree[current_node].children) == 0:
                node_path.pop()
                explored_children.pop()
                depth -= 1
                current_node = node_path[-1]
            #no more node to explore reach end of children
            elif len(nodes_tree[current_node].children) <= explored_children[depth]:
                node_path.pop()
                explored_children.pop()
                depth -= 1
                current_node = node_path[-1]
            #can explore a node
            else:
                next_children_index_node = explored_children[depth]
                next_node_index = nodes_tree[current_node].children[next_children_index_node]
                node_path.append(next_node_index)
                explored_children[depth] += 1
                explored_children.append(0)
                depth += 1
                current_node = node_path[-1]
        nodes_path.append(node_path)

    return nodes_path

def get_node_global_transform(gltf, node_number):
    node_path = []
    explored_children = [0]
    depth = -1
    nodes_tree = gltf.nodes
    current_node = 0

    node_roots = []

    for scene in gltf.scenes:
        node_roots = node_roots + scene.nodes

    while current_node != node_number:
        #root node reach without finding the target
        if depth == -1:
            next_root = node_roots.pop(0)
            if next_root == node_number:
                break
            elif not(nodes_tree[next_root].children):
                continue
            node_path.append(next_root)
            depth += 1
            current_node = next_root
        #leaf
        if len(nodes_tree[current_node].children) == 0:
            node_path.pop()
            explored_children.pop()
            depth -= 1
            current_node = node_path[-1]
        #no more node to explore reach end of children
        elif len(nodes_tree[current_node].children) <= explored_children[depth]:
            node_path.pop()
            explored_children.pop()
            depth -= 1
            current_node = node_path[-1]
        #can explore a node
        else:
            next_children_index_node = explored_children[depth]
            next_node_index = nodes_tree[current_node].children[next_children_index_node]
            node_path.append(next_node_index)
            explored_children[depth] += 1
            explored_children.append(0)
            depth += 1
            current_node = node_path[-1]

    transform = pyrr.Matrix44.identity()
    for i in range(depth+1):
        current_node_number = node_path[i]
        if gltf.nodes[current_node_number].matrix:
            next_transform = pyrr.Matrix44(np.array(gltf.nodes[current_node_number].matrix, dtype=np.float32).reshape(4, 4))
            transform = pyrr.matrix44.multiply(next_transform, transform)
        elif gltf.nodes[current_node_number].translation or gltf.nodes[current_node_number].rotation or gltf.nodes[current_node_number].scale:
            if gltf.nodes[current_node_number].translation:
                translation = pyrr.Matrix44.from_translation(gltf.nodes[current_node_number].translation, dtype=np.float32)
            else:
                translation = pyrr.Matrix44.identity(dtype=np.float32)

            if gltf.nodes[current_node_number].rotation:
                rotation = pyrr.Matrix44.from_quaternion(gltf.nodes[current_node_number].rotation, dtype=np.float32).transpose(1,0)
            else:
                rotation = pyrr.Matrix44.identity(dtype=np.float32)

            if gltf.nodes[current_node_number].scale:
                scale = pyrr.Matrix44.from_scale(gltf.nodes[current_node_number].scale, dtype=np.float32)
            else:
                scale = pyrr.Matrix44.identity(dtype=np.float32)

            next_transform = pyrr.matrix44.multiply(scale, rotation)
            next_transform = pyrr.matrix44.multiply(next_transform, translation)

            transform = pyrr.matrix44.multiply(next_transform, transform)

    return transform

def get_animation_node_global_transform(node_number, starting_node, nodes_transform, nodes_path):
    node_path = copy.copy(nodes_path[node_number])

    if starting_node:
        while node_path[0] != starting_node:
            node_path.pop(0)

    transform = pyrr.Matrix44.identity()
    for i in range(len(node_path)):
        current_node_number = node_path[i]
        if nodes_transform[current_node_number]["matrix"]:
            next_transform = np.array(nodes_transform[current_node_number]["matrix"], dtype=np.float32).reshape(4, 4)
            transform = pyrr.matrix44.multiply(next_transform, transform)
        elif nodes_transform[current_node_number]["translation"] or nodes_transform[current_node_number]["rotation"] or nodes_transform[current_node_number]["scale"]:
            if nodes_transform[current_node_number]["translation"]:
                translation = pyrr.Matrix44.from_translation(nodes_transform[current_node_number]["translation"], dtype=np.float32)
            else:
                translation = pyrr.Matrix44.identity(dtype=np.float32)

            if nodes_transform[current_node_number]["rotation"]:
                rotation = pyrr.Matrix44.from_quaternion(nodes_transform[current_node_number]["rotation"], dtype=np.float32).transpose(1,0)
            else:
                rotation = pyrr.Matrix44.identity(dtype=np.float32)

            if nodes_transform[current_node_number]["scale"]:
                scale = pyrr.Matrix44.from_scale(nodes_transform[current_node_number]["scale"], dtype=np.float32)
            else:
                scale = pyrr.Matrix44.identity(dtype=np.float32)

            next_transform = pyrr.matrix44.multiply(scale, rotation)
            next_transform = pyrr.matrix44.multiply(next_transform, translation)

            transform = pyrr.matrix44.multiply(next_transform, transform)

    return transform

def get_animation_node_global_transform_buffered(node_number, starting_node, nodes_transform, nodes_path, nodes_global_transform):
    node_path = copy.copy(nodes_path[node_number])

    if starting_node:
        while node_path[0] != starting_node:
            node_path.pop(0)

    global_transform_starting_indices = -1
    for i in range(len(node_path) - 1, 0, -1):
        if not isinstance(nodes_global_transform[node_path[i]], type(None)):
            global_transform_starting_indices = node_path[i-1]
            node_path = node_path[i:]
            break

    if global_transform_starting_indices == -1:
        transform = pyrr.Matrix44.identity()
    else:
        transform = nodes_global_transform[global_transform_starting_indices]
    for i in range(len(node_path)):
        current_node_number = node_path[i]
        if nodes_transform[current_node_number]["matrix"]:
            next_transform = pyrr.Matrix44(np.array(nodes_transform[current_node_number]["matrix"], dtype=np.float32).reshape(4, 4))
            transform = pyrr.matrix44.multiply(next_transform, transform)
        elif nodes_transform[current_node_number]["translation"] or nodes_transform[current_node_number]["rotation"] or nodes_transform[current_node_number]["scale"]:
            if nodes_transform[current_node_number]["translation"]:
                translation = pyrr.Matrix44.from_translation(nodes_transform[current_node_number]["translation"], dtype=np.float32)
            else:
                translation = pyrr.Matrix44.identity(dtype=np.float32)

            if nodes_transform[current_node_number]["rotation"]:
                rotation = pyrr.Matrix44.from_quaternion(nodes_transform[current_node_number]["rotation"], dtype=np.float32).transpose(1,0)
            else:
                rotation = pyrr.Matrix44.identity(dtype=np.float32)

            if nodes_transform[current_node_number]["scale"]:
                scale = pyrr.Matrix44.from_scale(nodes_transform[current_node_number]["scale"], dtype=np.float32)
            else:
                scale = pyrr.Matrix44.identity(dtype=np.float32)

            next_transform = pyrr.matrix44.multiply(scale, rotation)
            next_transform = pyrr.matrix44.multiply(next_transform, translation)

            transform = pyrr.matrix44.multiply(next_transform, transform)

        nodes_global_transform[node_path[i]] = transform

    return transform

def get_node_children(gltf, node_number):
    node_children = []
    nodes_tree = gltf.nodes
    node_to_explore = [] + nodes_tree[node_number].children

    while node_to_explore:
        current_node = node_to_explore.pop()
        node_children.append(current_node)

        if nodes_tree[current_node].children:
            node_to_explore = node_to_explore + nodes_tree[current_node].children

    return node_children

def generate_node_transform(nodes):
    nodes_transform = []

    for node_id, node in enumerate(nodes):
        nodes_transform.append({
            "id": node_id,
            "matrix":node.matrix,
            "rotation":node.rotation,
            "translation":node.translation,
            "scale":node.scale
        })

    return nodes_transform

def slerp(previous_quat, next_quat, interpolation_value):
    dot_product = np.dot(previous_quat,next_quat)

    #make sure we take the shortest path in case dot Product is negative
    if dot_product < 0.0:
        next_quat = -next_quat
        dot_product = -dot_product

    #if the two quaternions are too close to each other, just linear interpolate between the 4D vector
    if dot_product > 0.9995:
        transformation = previous_quat + interpolation_value * (next_quat - previous_quat)
        return transformation / np.linalg.norm(transformation)
    else:
        theta_0 = np.arccos(dot_product)
        theta = interpolation_value * theta_0
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        scale_previous_quat = np.cos(theta) - dot_product * sin_theta / sin_theta_0
        scale_next_quat = sin_theta / sin_theta_0

        return scale_previous_quat * previous_quat + scale_next_quat * next_quat

def load_gltf_model(root, gltf_file):
    gltf = GLTF2().load(gltf_file)

    meshes_indices = []
    meshes_items = []
    texture_ids = []

    nb_mesh = len(gltf.meshes)

    skin_index_for_mesh = []
    for mesh_number in range(nb_mesh):
        for node in gltf.nodes:
            if node.mesh == mesh_number:
                skin_index_for_mesh.append(node.skin)
                break

    nb_joint_per_skin = []
    for skin in gltf.skins:
        nb_joint_per_skin.append(len(skin.joints))

    usefull_joints_indices = []

    #retrieve the list of joint use to animate the mesh other joint are basically useless
    for current_mesh_number, mesh in enumerate(gltf.meshes):
        for primitive in mesh.primitives:
            if primitive.attributes.JOINTS_0:
                joint_accessor, joint_bufferView, joint_data, joint_str_converter, joint_offset, joint_nb_byte = get_accessor_data(gltf, primitive.attributes.JOINTS_0)

                for i in range(joint_accessor.count):
                    if primitive.attributes.JOINTS_0:
                        index = joint_bufferView.byteOffset + joint_accessor.byteOffset + i*joint_offset  # the location in the buffer of this vertex
                        d = joint_data[index:index+joint_nb_byte]  # the vertex data
                        v = struct.unpack(joint_str_converter, d)   # convert from base64 to three floats
                        for number in v:
                            number_joint_from_previous_skin = 0
                            for j in range(skin_index_for_mesh[current_mesh_number]):
                                number_joint_from_previous_skin += nb_joint_per_skin[j]

                            joint_indice = number + number_joint_from_previous_skin
                            if joint_indice not in usefull_joints_indices:
                                usefull_joints_indices.append(number + number_joint_from_previous_skin)
    usefull_joints_indices.sort()
    nb_usefull_joints = len(usefull_joints_indices)

    nb_items = []
    items_offset = []
    render_modes = []
    vertex_not_register = True
    texture_not_register = True
    texture_to_load = False
    normal_not_register = True
    normal_to_load = False
    joint_not_register = True
    joint_to_load = False

    #get all information about the model meshes: vertex, texture, normal, joint, weight, indices
    for current_mesh_number, mesh in enumerate(gltf.meshes):
        # get the vertices information for each primitive in the mesh
        for primitive in mesh.primitives:
            mesh_items = []
            mesh_indices = []
            # get the binary data for this mesh primitive from the buffer
            vertex_accessor, vertex_bufferView, vertex_data, vertex_str_converter, vertex_offset, vertex_nb_byte = get_accessor_data(gltf, primitive.attributes.POSITION)
            if vertex_not_register:
                nb_items.append(3)
                items_offset.append(0)
                vertex_not_register = not(vertex_not_register)

            if primitive.attributes.TEXCOORD_0:
                texture_accessor, texture_bufferView, texture_data, texture_str_converter, texture_offset, texture_nb_byte = get_accessor_data(gltf, primitive.attributes.TEXCOORD_0)
                if texture_not_register:
                    items_offset.append(items_offset[-1] + nb_items[-1] * 4)
                    nb_items.append(2)
                    texture_to_load = True
                    texture_not_register = not(texture_not_register)

            if primitive.attributes.NORMAL:
                normal_accessor, normal_bufferView, normal_data, normal_str_converter, normal_offset, normal_nb_byte = get_accessor_data(gltf, primitive.attributes.NORMAL)
                if normal_not_register:
                    items_offset.append(items_offset[-1] + nb_items[-1] * 4)
                    nb_items.append(3)
                    normal_to_load = True
                    normal_not_register = not(normal_not_register)

            if primitive.attributes.JOINTS_0:
                joint_accessor, joint_bufferView, joint_data, joint_str_converter, joint_offset, joint_nb_byte = get_accessor_data(gltf, primitive.attributes.JOINTS_0)
                if joint_not_register:
                    items_offset.append(items_offset[-1] + nb_items[-1] * 4)
                    nb_items.append(4)
                    joint_to_load = True

            if primitive.attributes.WEIGHTS_0:
                weight_accessor, weight_bufferView, weight_data, weight_str_converter, weight_offset, weight_nb_byte = get_accessor_data(gltf, primitive.attributes.WEIGHTS_0)
                if joint_not_register:
                    items_offset.append(items_offset[-1] + nb_items[-1] * 4)
                    nb_items.append(4)
                    joint_not_register = not(joint_not_register)

            indice_accessor, indice_bufferView, indice_data, indice_str_converter, indice_offset, indice_nb_byte = get_accessor_data(gltf, primitive.indices)

            # pull each information from the binary buffer and convert it into a tuple of python floats
            for i in range(vertex_accessor.count):
                #vertex
                index = vertex_bufferView.byteOffset + vertex_accessor.byteOffset + i*vertex_offset  # the location in the buffer of this vertex
                d = vertex_data[index:index+vertex_nb_byte]  # the vertex data
                v = struct.unpack(vertex_str_converter, d)   # convert from base64 to three floats
                for number in v:
                    mesh_items.append(number)
                #texture
                if primitive.attributes.TEXCOORD_0:
                    index = texture_bufferView.byteOffset + texture_accessor.byteOffset + i*texture_offset  # the location in the buffer of this texture
                    d = texture_data[index:index+texture_nb_byte]  # the texture data
                    v = struct.unpack(texture_str_converter, d)   # convert from base64 to three floats
                    for number in v:
                        mesh_items.append(number)
                #normal
                if primitive.attributes.NORMAL:
                    index = normal_bufferView.byteOffset + normal_accessor.byteOffset + i*normal_offset  # the location in the buffer of this normal
                    d = normal_data[index:index+normal_nb_byte]  # the normal data
                    v = struct.unpack(normal_str_converter, d)   # convert from base64 to three floats
                    for number in v:
                        mesh_items.append(number)
                #joint
                if primitive.attributes.JOINTS_0:
                    index = joint_bufferView.byteOffset + joint_accessor.byteOffset + i*joint_offset  # the location in the buffer of this joint
                    d = joint_data[index:index+joint_nb_byte]  # the joint data
                    v = struct.unpack(joint_str_converter, d)   # convert from base64 to three floats
                    for number in v:
                        number_joint_from_previous_skin = 0
                        for j in range(skin_index_for_mesh[current_mesh_number]):
                            number_joint_from_previous_skin += nb_joint_per_skin[j]
                        mesh_items.append(usefull_joints_indices.index(number + number_joint_from_previous_skin))

                #weight
                if primitive.attributes.WEIGHTS_0:
                    index = weight_bufferView.byteOffset + weight_accessor.byteOffset + i*weight_offset  # the location in the buffer of this weight
                    d = weight_data[index:index+weight_nb_byte]  # the weight data
                    v = struct.unpack(weight_str_converter, d)   # convert from base64 to three floats
                    for number in v:
                        mesh_items.append(number)

            meshes_items.append(mesh_items)

            for i in range(int(indice_accessor.count)):
                index = indice_bufferView.byteOffset + indice_accessor.byteOffset + i*indice_offset  # the location in the buffer of this index
                d = indice_data[index:index+indice_nb_byte]  # the index data
                v = struct.unpack(indice_str_converter, d)   # convert from base64 to three unsigned short
                for number in v:
                    mesh_indices.append(number)
            meshes_indices.append(mesh_indices)

            if texture_to_load:
                if gltf.materials[primitive.material].pbrMetallicRoughness.baseColorTexture:
                    texture_ids.append(gltf.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index)
                else:
                    texture_ids.append(None)

            if primitive.mode:
                render_modes.append(primitive.mode)
            else:
                render_modes.append(4)

    #model setup
    #load the meshes in opengl
    nb_meshes = len(meshes_items)
    VAO = glGenVertexArrays(nb_meshes)
    VBO = glGenBuffers(nb_meshes)
    EBO = glGenBuffers(nb_meshes)
    textures_buff = glGenTextures(nb_meshes)
    if nb_meshes == 1:
        VAO = np.array((VAO,))
        VBO = np.array((VBO,))
        EBO = np.array((EBO,))
        textures_buff = np.array((textures_buff,))

    mesh_number = 0

    for items, indices in zip(meshes_items, meshes_indices):
        # convert a numpy array for some manipulation
        np_indices = np.array(indices, dtype=np.uint32)

        np_items = np.array(items, dtype=np.float32)

        # Vertex Buffer Object
        glBindVertexArray(VAO[mesh_number])
        glBindBuffer(GL_ARRAY_BUFFER, VBO[mesh_number])
        glBufferData(GL_ARRAY_BUFFER, np_items.nbytes, np_items, GL_STATIC_DRAW)

        # Element Buffer Object
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[mesh_number])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np_indices.nbytes, np_indices, GL_STATIC_DRAW)

        nb_different_items = 0

        #vertices
        glVertexAttribPointer(nb_different_items, nb_items[nb_different_items], GL_FLOAT, GL_FALSE, np_items.itemsize * sum(nb_items), ctypes.c_void_p(items_offset[nb_different_items]))
        glEnableVertexAttribArray(nb_different_items)
        nb_different_items += 1

        if texture_to_load:
            #textures
            glVertexAttribPointer(nb_different_items, nb_items[nb_different_items], GL_FLOAT, GL_FALSE, np_items.itemsize * sum(nb_items), ctypes.c_void_p(items_offset[nb_different_items]))
            glEnableVertexAttribArray(nb_different_items)
            nb_different_items += 1

        if normal_to_load:
            #normals
            glVertexAttribPointer(nb_different_items, nb_items[nb_different_items], GL_FLOAT, GL_FALSE, np_items.itemsize * sum(nb_items), ctypes.c_void_p(items_offset[nb_different_items]))
            glEnableVertexAttribArray(nb_different_items)
            nb_different_items += 1

        if joint_to_load:
            #joints
            glVertexAttribPointer(nb_different_items, nb_items[nb_different_items], GL_FLOAT, GL_FALSE, np_items.itemsize * sum(nb_items), ctypes.c_void_p(items_offset[nb_different_items]))
            glEnableVertexAttribArray(nb_different_items)
            nb_different_items += 1

            #weights
            glVertexAttribPointer(nb_different_items, nb_items[nb_different_items], GL_FLOAT, GL_FALSE, np_items.itemsize * sum(nb_items), ctypes.c_void_p(items_offset[nb_different_items]))
            glEnableVertexAttribArray(nb_different_items)
            nb_different_items += 1

        mesh_number += 1

    if texture_to_load:
        for mesh_number, texture_id in enumerate(texture_ids):
            if isinstance(texture_id,int):
                load_texture(root + gltf.images[gltf.textures[texture_id].source].uri, textures_buff[mesh_number], gltf.samplers[gltf.textures[texture_id].sampler])

    #load the skin
    nodes_transform = None
    nodes_path = None
    joint_ids = None
    joint_matrices = None
    inverse_bind_matrices = None
    animations_data = None
    if joint_to_load:
        inverse_bind_matrices = []

        for skin in gltf.skins:
            inverseBindMatrixForJoint_accessor, inverseBindMatrixForJoint_bufferView, inverseBindMatrixForJoint_data, inverseBindMatrixForJoint_str_converter, inverseBindMatrixForJoint_offset, inverseBindMatrixForJoint_nb_byte = get_accessor_data(gltf, skin.inverseBindMatrices)

            for usefull_joints_indice in usefull_joints_indices:
                index = inverseBindMatrixForJoint_bufferView.byteOffset + inverseBindMatrixForJoint_accessor.byteOffset + usefull_joints_indice * inverseBindMatrixForJoint_offset  # the location in the buffer of this inverseBindMatrixForJoint
                d = inverseBindMatrixForJoint_data[index:index+inverseBindMatrixForJoint_nb_byte]  # the inverseBindMatrixForJoint data
                v = struct.unpack(inverseBindMatrixForJoint_str_converter, d)   # convert from base64 to three floats
                for number in v:
                    inverse_bind_matrices.append(number)

        inverse_bind_matrices = np.array(inverse_bind_matrices, dtype=np.float32).reshape(nb_usefull_joints, 4, 4)

        joint_matrices = []

        nodes_transform = generate_node_transform(gltf.nodes)
        nodes_path = generate_node_pass(gltf)

        joint_ids = []
        for skin in gltf.skins:
            joint_ids = joint_ids + skin.joints

        for indice, usefull_joints_indice in enumerate(usefull_joints_indices):
            global_joint_transform = get_node_global_transform(gltf, joint_ids[usefull_joints_indice])
            joint_matrices.append(pyrr.matrix44.multiply(inverse_bind_matrices[indice], global_joint_transform))

        joint_matrices = np.array(joint_matrices, dtype=np.float32).reshape(nb_usefull_joints, 4, 4)

        #load animations
        animations_data = []
        for animation in gltf.animations:
            animation_data = []

            for channel in animation.channels:
                sampler_id = channel.sampler
                node_id = channel.target.node
                type_of_transformation = channel.target.path
                interpolation_type = animation.samplers[sampler_id].interpolation

                animation_timings = []
                timing_accessor, timing_bufferView, timing_data, timing_str_converter, timing_offset, timing_nb_byte = get_accessor_data(gltf, animation.samplers[sampler_id].input)

                # pull each vertex and indice from the binary buffer and convert it into a tuple of python floats
                for i in range(timing_accessor.count):
                    index = timing_bufferView.byteOffset + timing_accessor.byteOffset + i*timing_offset  # the location in the buffer of this timing
                    d = timing_data[index:index+timing_nb_byte]  # the timing data
                    v = struct.unpack(timing_str_converter, d)   # convert from base64 to three floats
                    for number in v:
                        animation_timings.append(number)

                animation_values = []
                value_accessor, value_bufferView, value_data, value_str_converter, value_offset, value_nb_byte = get_accessor_data(gltf, animation.samplers[sampler_id].output)

                # pull each vertex and indice from the binary buffer and convert it into a tuple of python floats
                for i in range(value_accessor.count):
                    index = value_bufferView.byteOffset + value_accessor.byteOffset + i*value_offset  # the location in the buffer of this value
                    d = value_data[index:index+value_nb_byte]  # the value data
                    v = struct.unpack(value_str_converter, d)   # convert from base64 to three floats
                    animation_values.append(v)

                animation_data.append({"node_id":node_id,
                                    "transformation_type":type_of_transformation,
                                    "interpolation_type":interpolation_type,
                                    "times":animation_timings,
                                        "values": animation_values})

            animations_data.append({"name":animation.name,
                                    "data":animation_data})

    nb_nodes = len(gltf.nodes)

    return texture_to_load, normal_to_load, joint_to_load, nb_meshes, render_modes, meshes_indices, nb_nodes, nodes_transform, nodes_path, nb_usefull_joints, usefull_joints_indices, joint_ids, joint_matrices, inverse_bind_matrices, animations_data, VAO, texture_ids, textures_buff

def load_background(background_texture_file, background_shader):
    glUseProgram(background_shader)

    vao_background = glGenVertexArrays(1)
    vbo_background = glGenBuffers(1)
    texture_background = glGenTextures(1)

    #vertex, texture
    background_vertices = [
        -1, 1,  0, 0,
        1, 1,   1, 0,
        1, -1,  1, 1,
        -1, 1,  0, 0,
        -1, -1, 0, 1,
        1, -1,  1, 1
    ]
    np_background_vertices = np.array(background_vertices, dtype=np.float32)

    glBindVertexArray(vao_background)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_background)
    glBufferData(GL_ARRAY_BUFFER, np_background_vertices.nbytes, np_background_vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, np_background_vertices.itemsize * 4, ctypes.c_void_p(0)) #vertex is describe by 2 float, each vertex have 4 float data(2 coordinate, 2 texture coordinate), offset starting at 0 byte
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, np_background_vertices.itemsize * 4, ctypes.c_void_p(8)) #texture is describe by 2 float, each vertex have 4 float data(2 coordinate, 2 texture coordinate), offset starting at 8 byte(2 float(4 byte))
    glEnableVertexAttribArray(1)

    glBindTexture(GL_TEXTURE_2D, texture_background)

    image = Image.open(background_texture_file)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glGenerateMipmap(GL_TEXTURE_2D)

    return vao_background, texture_background

def draw_background(background_shader, vao_background, texture_background):
    glUseProgram(background_shader)
    glDisable(GL_DEPTH_TEST)
    glBindVertexArray(vao_background)
    glBindTexture(GL_TEXTURE_2D, texture_background)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glEnable(GL_DEPTH_TEST)

def draw_model(shader, ct, total_animation_time, model_loc, model_pos, texture_to_load, joint_to_load, nb_meshes, render_modes, meshes_indices, nodes_transform, nodes_path, nb_usefull_joints, usefull_joints_indices, joint_ids, joint_matrix, joint_matrices, inverse_bind_matrices, nodes_global_transform, animation_number, animations_data, VAO, texture_ids, textures_buff):
    glUseProgram(shader)

    rot_x = pyrr.Matrix44.from_x_rotation(0.5 * ct)
    rot_y = pyrr.Matrix44.from_y_rotation(0.8 * ct)
    model = pyrr.matrix44.multiply(rot_y, model_pos)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    #animation
    if joint_to_load:
        if animations_data:
            current_time = ct % total_animation_time
            for animation in animations_data[animation_number]["data"]:
                indices = 0
                while current_time > animation["times"][indices]:
                    indices += 1
                    if indices == len(animation["times"]):
                        indices -= 1
                        break
                indices -= 1

                if animation["interpolation_type"] == "STEP":
                    if len(animation["times"]) == 1:
                        nodes_transform[animation["node_id"]][animation["transformation_type"]] = list(animation["values"][0])
                    else:
                        transformation = animation["values"][indices]
                        nodes_transform[animation["node_id"]][animation["transformation_type"]] = list(transformation)

                elif animation["interpolation_type"] == "LINEAR":
                    if len(animation["times"]) == 1:
                        nodes_transform[animation["node_id"]][animation["transformation_type"]] = list(animation["values"][0])
                    else:
                        transformation = np.array(animation["values"][indices])
                        transformation2 = np.array(animation["values"][(indices+1)])

                        interpolation_value = (current_time - animation["times"][indices]) / (animation["times"][(indices+1)] - animation["times"][indices])

                        if animation["transformation_type"] == "rotation":
                            transformation = slerp(transformation, transformation2, interpolation_value)
                            nodes_transform[animation["node_id"]]["rotation"] = list(transformation)
                        else:
                            transformation = transformation + interpolation_value * (transformation2 - transformation)
                            nodes_transform[animation["node_id"]][animation["transformation_type"]] = list(transformation)

                elif animation["interpolation_type"] == "CUBICSPLINE":
                    print("CUBICSPLINE interpolation mode not implemented.", file = sys.stderr)

                nodes_global_transform[animation["node_id"]] = None

            for indice, usefull_joints_indice in enumerate(usefull_joints_indices):
                global_joint_transform = get_animation_node_global_transform_buffered(joint_ids[usefull_joints_indice], None, nodes_transform, nodes_path, nodes_global_transform)
                joint_matrices[indice] = pyrr.matrix44.multiply(inverse_bind_matrices[indice], global_joint_transform)

            glUniformMatrix4fv(joint_matrix, nb_usefull_joints, GL_FALSE, joint_matrices)

    for i in range(nb_meshes):
        glBindVertexArray(VAO[i])
        if texture_to_load:
            if isinstance(texture_ids[i], int):
                glBindTexture(GL_TEXTURE_2D, textures_buff[i])

        glDrawElements(gltf_render_mode[str(render_modes[i])], len(meshes_indices[i]), GL_UNSIGNED_INT, None)

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE)

    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, 16 / 9, 0.1, 1000)
    # eye, target, up
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0.5, 2]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

    background_file = "./assets/test/chess.png"
    background_shader = compileProgram(compileShader(background_vertex_src, GL_VERTEX_SHADER), compileShader(background_fragment_src, GL_FRAGMENT_SHADER))
    #background texture setup
    vao_background, texture_background = load_background(background_file, background_shader)

    # Experimental 3D rendering is disabled in the shipped pipeline (AVAILABLE_MODELS
    # is 2D only). Point ``root`` at a folder containing a glTF model to try it.
    root = "./assets/3d_models/example/"
    gltf_filename = root + "scene.gltf"

    texture_to_load, normal_to_load, joint_to_load, nb_meshes, render_modes, meshes_indices, nb_nodes, nodes_transform, nodes_path, nb_usefull_joints, usefull_joints_indices, joint_ids, joint_matrices, inverse_bind_matrices, animations_data, VAO, texture_ids, textures_buff = load_gltf_model(root, gltf_filename)

    shader = compile_shader(texture_to_load, normal_to_load, joint_to_load, nb_usefull_joints)

    #configure shader
    glUseProgram(shader)

    model_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
    zoom = 1
    zoom_matrice = pyrr.Matrix44.from_scale([zoom]*3, dtype=np.float32)

    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    zoom_scale = glGetUniformLocation(shader, "zoom")
    if joint_to_load:
        joint_matrix = glGetUniformLocation(shader, "u_jointMatrix")
    else:
        joint_matrix = None

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(zoom_scale, 1, GL_FALSE, zoom_matrice)

    #load the skin
    if joint_to_load:
        glUniformMatrix4fv(joint_matrix, nb_usefull_joints, GL_FALSE, joint_matrices)

        if animations_data:
            animation_number = 0

            total_animation_time = 0
            for animation in animations_data[animation_number]["data"]:
                if total_animation_time < animation["times"][-1]:
                    total_animation_time = animation["times"][-1]
    else:
        animation_number = 0
        total_animation_time = 0

    running = True

    frame_array_len = 100
    frame_time = np.zeros((frame_array_len))
    frame_indices = 0

    nodes_global_transform = [None] * nb_nodes

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                glViewport(0, 0, event.w, event.h)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_background(background_shader, vao_background, texture_background)

        ct = pygame.time.get_ticks() / 1000

        draw_model(shader, ct, total_animation_time, model_loc, model_pos, texture_to_load, joint_to_load, nb_meshes, render_modes, meshes_indices, nodes_transform, nodes_path, nb_usefull_joints, usefull_joints_indices, joint_ids, joint_matrix, joint_matrices, inverse_bind_matrices, nodes_global_transform, animation_number, animations_data, VAO, texture_ids, textures_buff)

        ct_end = pygame.time.get_ticks() / 1000
        frame_time[frame_indices] = ct_end - ct
        fps = 1 / np.mean(frame_time)
        frame_indices = (frame_indices + 1) % frame_array_len
        print(fps)

        pygame.display.flip()

    pygame.quit()
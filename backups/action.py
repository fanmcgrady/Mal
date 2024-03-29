# 需要修改

# encoding=utf-8
# * modify exports using lief
# * zero out rich header (if it exists) -->
# requires updating OptionalHeader's checksum ("Rich Header" only in Microsoft-produced executables)
# * tinker with resources: https://lief.quarkslab.com/doc/tutorials/07_pe_resource.html
# also in our project dir. : /test/lief-tutorials/PE_resource

import array
import json
import os
import random
import struct  # byte manipulations
import sys

import lief  # pip install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz

from gym_malware.envs.utils import interface

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

COMMON_SECTION_NAMES = open(os.path.join(
    module_path, 'section_names.txt'), 'r').read().rstrip().split('\n')
COMMON_IMPORTS = json.load(
    open(os.path.join(module_path, 'small_dll_imports.json'), 'r'))

######################
# explicitly list so that these may be used externally
ACTION_TABLE = {
    'ARBE': 'ARBE',
    'imports_append': 'imports_append',
    'random_imports_append': 'random_imports_append',
    'ARS': 'ARS',
    'imports_append2': 'imports_append2',
    'section_rename': 'section_rename',
    'section_append': 'section_append',
    'create_new_entry': 'create_new_entry'
}


# action 操作类
class MalwareManipulator():
    def __init__(self, bytez):
        self.bytez = bytez
        self.min_append_log2 = 5
        self.max_append_log2 = 8

    # 构造随机长度
    def __random_length(self):
        return 2 ** random.randint(self.min_append_log2, self.max_append_log2)

    # 判断是否已包含某库
    def __has_random_lib(self, imports, lowerlibname):
        for im in imports:
            if im.name.lower() == lowerlibname:
                return True
        return False

    # 把lief结果build成bytez
    def __binary_to_bytez(self, binary, dos_stub=False, imports=False, overlay=False, relocations=False,
                          resources=False, tls=False):
        # write the file back as bytez
        builder = lief.PE.Builder(binary)
        builder.build_dos_stub(dos_stub)  # rebuild DOS stub

        builder.build_imports(imports)  # rebuild IAT in another section
        builder.patch_imports(imports)  # patch original import table with trampolines to new import table

        builder.build_overlay(overlay)  # rebuild overlay
        builder.build_relocations(relocations)  # rebuild relocation table in another section
        builder.build_resources(resources)  # rebuild resources in another section
        builder.build_tls(tls)  # rebuilt TLS object in another section

        builder.build()  # perform the build process

        # return bytestring
        return array.array('B', builder.get_build()).tobytes()

    # 生成随机的import name
    def __generate_random_import_libname(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        suffix = random.choice(['.dll', '.exe'])
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length)) + suffix

    # 生成随机函数名
    def __generate_random_name(self, minlength=5, maxlength=7):
        length = random.randint(minlength, maxlength)
        return "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(length))

    # action 1: append bytes to the overlay (end of PE file)
    def ARBE(self, seed=None):  # random加的？？？
        random.seed(seed)
        L = self.__random_length()
        # choose the upper bound for a uniform distribution in [0,upper]
        upper = random.randrange(256)
        # upper chooses the upper bound on uniform distribution:
        # upper=0 would append with all 0s
        # upper=126 would append with "printable ascii"
        # upper=255 would append with any character
        return self.bytez + bytes([random.randint(0, upper) for _ in range(L)])

    # action 2: add a function to the import address table that is never used
    def imports_append(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.parse(self.bytez)

        importslist = binary.imports
        # draw a library at random
        libname = random.choice(list(COMMON_IMPORTS.keys()))
        funcname = random.choice(list(COMMON_IMPORTS[libname]))
        lowerlibname = libname.lower()

        count_limit = 0

        while self.__has_random_lib(importslist, lowerlibname):
            # draw a library at random
            libname = random.choice(list(COMMON_IMPORTS.keys()))
            funcname = random.choice(list(COMMON_IMPORTS[libname]))
            lowerlibname = libname.lower()
            count_limit += 1
            if count_limit > 10:
                break

        # add a new library
        lib = binary.add_library(libname)

        # get current names
        names = set([e.name for e in lib.entries])
        if not funcname in names:
            lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

    # action 3: add a function to the import address table that is random name
    def random_imports_append(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # draw a library at random
        libname = self.__generate_random_import_libname()
        funcname = self.__generate_random_name()
        lowerlibname = libname.lower()
        # append this lib in the imports
        lib = binary.add_library(lowerlibname)
        lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

    # action 4: create a new(unused) sections
    def ARS(self, seed=None):
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # 建立一个section
        new_section = lief.PE.Section(self.__generate_random_name())

        # fill with random content
        upper = random.randrange(256)  # section含content、虚拟地址、type
        L = self.__random_length()
        new_section.content = [random.randint(0, upper) for _ in range(L)]

        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])

        # add a new empty section
        binary.add_section(new_section,
                           random.choice([
                               lief.PE.SECTION_TYPES.BSS,
                               lief.PE.SECTION_TYPES.DATA,
                               lief.PE.SECTION_TYPES.EXPORT,
                               lief.PE.SECTION_TYPES.IDATA,
                               lief.PE.SECTION_TYPES.RELOCATION,
                               lief.PE.SECTION_TYPES.RESOURCE,
                               lief.PE.SECTION_TYPES.TEXT,
                               lief.PE.SECTION_TYPES.TLS_,
                               lief.PE.SECTION_TYPES.UNKNOWN,
                           ]))

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # action 5: add a function to the import address table that is never used
    def imports_append2(self, seed=None):
        # add (unused) imports
        random.seed(seed)
        binary = lief.parse(self.bytez)
        # draw a library at random
        libname = random.choice(list(COMMON_IMPORTS.keys()))  # 随机选择？
        funcname = random.choice(list(COMMON_IMPORTS[libname]))  # 随机选择？
        lowerlibname = libname.lower()
        # find this lib in the imports, if it exists
        lib = None
        for im in binary.imports:
            if im.name.lower() == lowerlibname:
                lib = im
                break
        if lib is None:
            # add a new library
            lib = binary.add_library(libname)
        # get current names
        names = set([e.name for e in lib.entries])  # 一个lib + lib里的entry
        if not funcname in names:
            lib.add_entry(funcname)

        self.bytez = self.__binary_to_bytez(binary, imports=True)

        return self.bytez

    # action 6: manipulate existing section names
    def section_rename(self, seed=None):
        # rename a random section
        random.seed(seed)
        binary = lief.parse(self.bytez)
        targeted_section = random.choice(binary.sections)
        targeted_section.name = random.choice(COMMON_SECTION_NAMES)[:7]  # current version of lief not allowing 8 chars?

        self.bytez = self.__binary_to_bytez(binary)

        return self.bytez

    # TODO: lief接口问题
    # action 7: append bytes to extra space at the end of sections
    def section_append(self, seed=None):
        # append to a section (changes size and entropy)
        random.seed(seed)
        binary = lief.parse(self.bytez)
        targeted_section = random.choice(binary.sections)
        L = self.__random_length()
        print("targeted_section.size:",targeted_section.size)
        print("targeted_section.content:",type(targeted_section.content))
        available_size = targeted_section.size - len(targeted_section.content)
        print("available_size:{}".format(available_size))
        if L > available_size:
            L = available_size

        upper = random.randrange(256)
        # targeted_section.content = targeted_section.content + \
        #                            [random.randint(0, upper) for _ in range(L)]
        for i in range(L):
            targeted_section.content[L+i] = os.urandom(1)
        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez

    # action 8: create a new entry point which immediately jumps to the original entry point
    def create_new_entry(self, seed=None):
        # create a new section with jump to old entry point, and change entry point
        # DRAFT: this may have a few technical issues with it (not accounting for relocations),
        # but is a proof of concept for functionality
        random.seed(seed)

        binary = lief.parse(self.bytez)

        # get entry point
        entry_point = binary.optional_header.addressof_entrypoint

        # get name of section
        entryname = binary.section_from_rva(entry_point).name

        # create a new section
        new_section = lief.PE.Section(entryname + "".join(chr(random.randrange(
            ord('.'), ord('z'))) for _ in range(3)))  # e.g., ".text" + 3 random characters
        # push [old_entry_point]; ret
        new_section.content = [
                                  0x68] + list(struct.pack("<I", entry_point + 0x10000)) + [0xc3]
        new_section.virtual_address = max(
            [s.virtual_address + s.size for s in binary.sections])
        # TO DO: account for base relocation (this is just a proof of concepts)

        # add new section
        binary.add_section(new_section, lief.PE.SECTION_TYPES.TEXT)

        # redirect entry point
        binary.optional_header.addressof_entrypoint = new_section.virtual_address

        self.bytez = self.__binary_to_bytez(binary)
        return self.bytez


def modify_without_breaking(bytez, action=None, seed=None):
    _action = MalwareManipulator(bytez).__getattribute__(ACTION_TABLE[action])

    try:
        bytez = _action(seed)
    except Exception as e:  # ..then become petulant
        print(e)
        print('action error ')

    import hashlib
    m = hashlib.sha256()
    m.update(bytez)
    return bytez


def test_ARBE(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.ARBE()
    binary2 = lief.parse(bytez2)
    if len(binary.overlay) == len(binary2.overlay):
        return 0
    else:
        return 1


def test_imports_append(bytez):
    binary = lief.parse(bytez)
    # SUCCEEDS, but note that lief builder also adds a new ".l1" section for each patch of the imports
    manip = MalwareManipulator(bytez)
    bytez2 = manip.imports_append(bytez)
    # bytez2 = manip.imports_append_org(bytez)
    binary2 = lief.parse(bytez2)
    # set1 = set(binary.imported_functions)
    # set2 = set(binary2.imported_functions)
    # diff = set2.difference(set1)
    # print(list(diff))
    if len(binary.imported_functions) == len(binary2.imported_functions):
        return 0
    else:
        return 1


def test_random_imports_append(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.imports_append(bytez)
    binary2 = lief.parse(bytez2)
    if len(binary.imported_functions) == len(binary2.imported_functions):
        return 0
    else:
        return 1


def test_ARS(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.ARS(bytez)
    binary2 = lief.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    if len(newsections) == len(oldsections):
        return 0
    else:
        return 1


def test_imports_append2(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.imports_append2(bytez)
    binary2 = lief.parse(bytez2)
    if len(binary.imported_functions) == len(binary2.imported_functions):
        return 0
    else:
        return 1


def test_section_rename(bytez):
    binary = lief.parse(bytez)
    # SUCCEEDS
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_rename(bytez)
    binary2 = lief.parse(bytez2)
    oldsections = [s.name for s in binary.sections]
    newsections = [s.name for s in binary2.sections]
    # print(oldsections)
    # print(newsections)
    if " ".join(newsections) == " ".join(oldsections):
        return 0
    else:
        return 1


def test_section_append(bytez):
    binary = lief.parse(bytez)
    # FAILS if there's insufficient room to add to the section
    manip = MalwareManipulator(bytez)
    bytez2 = manip.section_append(bytez)
    binary2 = lief.parse(bytez2)
    oldsections = [len(s.content) for s in binary.sections]
    newsections = [len(s.content) for s in binary2.sections]
    print(oldsections)
    print(newsections)
    if sum(newsections) == sum(oldsections):
        return 0
    else:
        return 1


def test_create_new_entry(bytez):
    binary = lief.parse(bytez)
    manip = MalwareManipulator(bytez)
    bytez2 = manip.create_new_entry(bytez)
    binary2 = lief.parse(bytez2)
    # print(binary.entrypoint)
    # print(binary2.entrypoint)
    if binary.entrypoint == binary2.entrypoint:
        return 0
    else:
        return 1


if __name__ == '__main__':
    bytez = interface.fetch_file("Exploit.Win32.ActivePost.g")
    # print(test_ARBE(bytez))
    # print(test_imports_append(bytez))
    # print(test_random_imports_append(bytez))
    # print(test_ARS(bytez))
    # print(test_imports_append2(bytez))
    # print(test_section_rename(bytez))
    print(test_section_append(bytez))
    # print(test_create_new_entry(bytez))

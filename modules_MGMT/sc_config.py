from os.path import join


def masterpro_vars():
    """
    return: seiscomp masterpro host, user, pass, and db name
    """
    return ''


def clientpro_vars():
    """
    return: seiscomp clientpro host, user, pass, and db name
    """
    return ''


def seiscomp_dirs():
    """
    return: seiscomp archive and inventory directory
    """
    # arc_dir = join('/home', 'data', 'archive')
    #inv_dir = join('/home', 'sysop', 'seiscomp3', 'etc', 'inventory')
    arc_dir = join('/home', 'sysop', 'seiscomp', 'var', 'lib', 'archive')
    inv_dir = join('/home', 'sysop', 'seiscomp', 'etc', 'inventory')
    # arc_dir = join('/home', 'eq', 'PycharmProjects', 'Q_RMT', 'archive_data')
    # inv_dir = join('/home', 'eq', 'PycharmProjects', 'Q_RMT', 'inventory')
    return arc_dir, inv_dir


def bmkg_seedlink_vars():
    """
    return: bmkg seedlink host, port
    """
    return '', 18000


def seedlink_vars():
    """
    return: pgr seedlink host, port
    """
    return '', 18000


def url_index3():
    """
    return: url index3 seiscomp PGN
    """
    return ''


def url_BMKG_API ():
    """
    return: url BMKG data online
    """
    return ''


def url_WRS_API():
    """
    return: url API WRS NewGen
    """
    return ''


def facebook_token():
    """
    :return: facebook access token
    """
    return ''


def ftp_server():
    """
    :return: stageof ambon ftp server account
    """
    return '182.16.xx.xx', 3042, 'hosting', '*****************'
